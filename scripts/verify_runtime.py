#!/usr/bin/env python3
"""Verify the owned server's configuration and memory during a short SSE request."""
import argparse
import json
from benchmark import api_headers
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=ROOT/'logs/service-final-runtime.json')
args=parser.parse_args()
state=json.loads((ROOT/'state/server.json').read_text())
process=Path('/proc')/str(state['pid'])
url=state['url']
with urlopen(url+'/health',timeout=10) as response:health=json.load(response)
with urlopen(Request(url+'/props',headers=api_headers()),timeout=10) as response:props=json.load(response)
with urlopen(url+'/',timeout=10) as response:ui={'status':response.status,'has_html':b'DeepSeek Flash' in response.read()}
ui['assets']={}
for name,mime in [('app.css','text/css'),('app.js','text/javascript'),('metrics.js','text/javascript'),('settings.js','text/javascript'),('benchmark.json','application/json')]:
    with urlopen(url+'/'+name,timeout=10) as response:
        ui['assets'][name]={'status':response.status,'content_type':response.headers.get('Content-Type'),'bytes':len(response.read())}
    assert ui['assets'][name]['status']==200 and ui['assets'][name]['bytes']>0 and ui['assets'][name]['content_type'].startswith(mime),ui['assets'][name]

def memory():
    return {line.split(':',1)[0]:int(line.split()[1])*1024 for line in (process/'status').read_text().splitlines() if line.startswith(('VmRSS:','VmSwap:','VmLck:'))}
def counters():
    io={line.split(':')[0]:int(line.split(':')[1]) for line in (process/'io').read_text().splitlines()}
    return io['read_bytes'],int((process/'stat').read_text().rsplit(') ',1)[1].split()[9])

vram=max(Path('/sys/class/drm').glob('card*/device/mem_info_vram_total'),key=lambda p:int(p.read_text())).parent/'mem_info_vram_used'
result={'pid':state['pid'],'health':health,'ui':ui,'engine_options':state['engine_options'],'command':state['command'],'context':props['default_generation_settings']['n_ctx']}
for line in (process/'cgroup').read_text().splitlines():
    if line.startswith('0::'):
        group=Path('/sys/fs/cgroup')/line[3:].lstrip('/')
        result['memory_swap_max']=(group/'memory.swap.max').read_text().strip()
        result['memory_swap_current']=(group/'memory.swap.current').read_text().strip()
stream={'done':False,'content':'','reasoning_content':'','native_samples':0}
def generate():
    try:
        body={'model':'dsv4f','messages':[{'role':'user','content':'Explain story of the romance of three kingdoms.'}], 'max_tokens':64,'temperature':0,'seed':42,'cache_prompt':False,'stream':True,'stream_options':{'include_usage':True},'timings_per_token':True,'reasoning_effort':'none','reasoning_budget_tokens':0,'chat_template_kwargs':{'enable_thinking':False}}
        with urlopen(Request(url+'/v1/chat/completions',data=json.dumps(body).encode(),headers=api_headers()),timeout=180) as response:
            for line in response:
                if not line.startswith(b'data:'):continue
                value=line[5:].strip()
                if value==b'[DONE]':stream['done']=True;break
                if not value:continue
                chunk=json.loads(value)
                if chunk.get('error'):raise RuntimeError(chunk['error'])
                delta=(chunk.get('choices') or [{}])[0].get('delta',{})
                for key in ['content','reasoning_content']:stream[key]+=delta.get(key) or ''
                if chunk.get('usage'):stream['usage']=chunk['usage']
                if chunk.get('timings'):stream['timings']=chunk['timings'];stream['native_samples']+=1
    except Exception as exc:stream['error']=str(exc)
start=time.monotonic();before=counters();samples=[]
worker=threading.Thread(target=generate);worker.start()
while worker.is_alive():
    samples.append(int(vram.read_text()));worker.join(.1)
after=counters()
result.update(memory=memory(),observation_seconds=time.monotonic()-start,read_bytes_delta=after[0]-before[0],major_fault_delta=after[1]-before[1],vram_gib_min=min(samples)/2**30,vram_gib_max=max(samples)/2**30,stream_smoke=stream)
args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
assert health['status']=='ok' and result['context']==65536
assert result['memory']['VmSwap']==0 and result['memory_swap_max']=='0' and result['memory_swap_current']=='0'
assert '--cache-type-k' in result['command'] and result['command'][result['command'].index('--cache-type-k')+1]=='q8_0'
assert result['command'][result['command'].index('--cache-type-v')+1]=='q8_0'
assert stream['done'] and stream['content'] and not stream['reasoning_content'] and stream['native_samples']>1,stream
assert result['read_bytes_delta']==0 and result['major_fault_delta']==0,result
print(json.dumps({k:result[k] for k in ['health','context','memory','vram_gib_min','vram_gib_max','read_bytes_delta','major_fault_delta','memory_swap_max']},ensure_ascii=False))
