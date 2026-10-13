#!/usr/bin/env python3
"""Native decode benchmark plus read-only AMDGPU samples during each request."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import threading
import time
from urllib.request import Request, urlopen
from benchmark import generate, PROMPT, api_headers

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--tokens',type=int,default=128);p.add_argument('--runs',type=int,default=3);p.add_argument('--warmup-tokens',type=int,default=32);p.add_argument('--url',default='http://127.0.0.1:8080');a=p.parse_args()
    devices=list(Path('/sys/class/drm').glob('card*/device/mem_info_vram_total'));device=next((x.parent for x in devices if int(x.read_text())>0),None)
    hwmon=next(iter((device/'hwmon').glob('hwmon*')),None) if device else None
    nodes={'gpu_busy_percent':device/'gpu_busy_percent','mem_busy_percent':device/'mem_busy_percent','vram_bytes':device/'mem_info_vram_used'} if device else {}
    manifest=Path(__file__).resolve().parents[1]/'state'/'server.json'
    engine_state=json.loads(manifest.read_text()) if manifest.exists() else {}
    proc=Path('/proc')/str(engine_state['pid']) if engine_state.get('url')==a.url else None
    ticks_per_second=os.sysconf('SC_CLK_TCK');logical_cpus=os.cpu_count() or 1
    if hwmon:
        for name in ['freq1_input','freq2_input','power1_average','power1_input']:nodes[name]=hwmon/name
    samples=[];stop=threading.Event();stage='warmup'
    def sample():
        previous=None
        while not stop.is_set():
            row={'time':time.monotonic(),'stage':stage}
            if proc:
                try:
                    stat=(proc/'stat').read_text().rsplit(') ',1)[1].split()
                    ticks=int(stat[11])+int(stat[12])
                    if previous:
                        elapsed=row['time']-previous[0]
                        row['process_cpu_percent']=100*(ticks-previous[1])/ticks_per_second/elapsed/logical_cpus
                    previous=(row['time'],ticks)
                except (OSError,ValueError):pass
            for k,path in nodes.items():
                try:row[k]=int(path.read_text())
                except (OSError,ValueError):pass
            samples.append(row);stop.wait(.1)
    with urlopen(Request(a.url+'/props',headers=api_headers()),timeout=10) as r:props=json.load(r)
    t=threading.Thread(target=sample);t.start();results=[]
    try:
        if a.warmup_tokens:generate(a.url,a.warmup_tokens)
        for i in range(a.runs):
            stage='run'+str(i+1);result=generate(a.url,a.tokens);results.append(result)
            print(stage,round(result['timings']['predicted_per_second'],3),'tok/s',flush=True)
    finally:stop.set();t.join()
    summaries={}
    for name in dict.fromkeys(s['stage'] for s in samples):
        group=[s for s in samples if s['stage']==name];summary={}
        for key in [*nodes,'process_cpu_percent']:
            values=[s[key] for s in group if key in s]
            if values:summary[key]={'mean':statistics.mean(values),'median':statistics.median(values),'min':min(values),'max':max(values)}
        summaries[name]=summary
    engine_metadata={k:engine_state[k] for k in ('pid','log','engine_options') if k in engine_state} if proc else None
    report={'date_utc':datetime.now(timezone.utc).isoformat(),'prompt':PROMPT,'reasoning':False,'warmup_tokens':a.warmup_tokens,'props':props,'engine_metadata':engine_metadata,'runs':results,'median_decode_tokens_per_second':statistics.median(r['timings']['predicted_per_second'] for r in results),'hardware_summary':summaries,'hardware_samples':samples,'logical_cpus':logical_cpus,'process_cpu_note':'Process CPU time divided by wall time and logical CPU count; includes all server threads and prompt processing.','hardware_note':'100 ms sysfs samples during whole requests, including prompt processing; not a GPU kernel timeline.'}
    a.output.write_text(json.dumps(report,indent=2));print('median',report['median_decode_tokens_per_second'],'tok/s',flush=True)
if __name__=='__main__':main()
