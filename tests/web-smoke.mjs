// Component smoke test with a minimal DOM and IndexedDB API double. No browser automation.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
class Element {
  constructor(tag='div') {this.tagName=tag;this.children=[];this._text='';this.value='';this.files=[];this.attributes={};this.hidden=false;this.disabled=false;this.className='';this.style={setProperty:(k,v)=>this.style[k]=v};this.classList={toggle:(v,on)=>{const a=new Set(this.className.split(' ').filter(Boolean));const next=on??!a.has(v);if(next)a.add(v);else a.delete(v);this.className=[...a].join(' ');return next;},contains:v=>this.className.split(' ').includes(v)};}
  set textContent(v){this._text=String(v);this.children=[];}
  get textContent(){return this._text+this.children.map(c=>c.textContent).join('');}
  append(...items){for(let x of items){if(typeof x==='string'){const e=new Element('#text');e.textContent=x;x=e;}x.parent=this;this.children.push(x);}}
  replaceChildren(...items){this._text='';this.children=[];this.append(...items);}
  insertBefore(el,before){const i=this.children.indexOf(before);this.children.splice(i<0?this.children.length:i,0,el);}
  setAttribute(k,v){this.attributes[k]=v;}
  removeAttribute(k){delete this.attributes[k];}
  get childElementCount(){return this.children.length;}
  get lastElementChild(){return this.children.at(-1);}
  querySelector(selector){for(const c of this.children){if(selector.startsWith('.')&&c.className.split(' ').includes(selector.slice(1)))return c;const nested=c.querySelector(selector);if(nested)return nested;}return null;}
  addEventListener(){}
  focus(){}
  showModal(){this.open=true;}
  close(){this.open=false;this.onclose?.();}
  reportValidity(){return true;}
  setPointerCapture(id){this.pointerId=id;}
  hasPointerCapture(id){return this.pointerId===id;}
  releasePointerCapture(){this.pointerId=undefined;}
}
const html=await readFile(new URL('../web/index.html', import.meta.url),'utf8');
const disk=new Map();
globalThis.indexedDB={open(){const req={};queueMicrotask(()=>{req.result={transaction(){const tx={objectStore(){return {get(k){const r={};queueMicrotask(()=>{r.result=disk.get(k);tx.oncomplete?.();});return r;},put(v,k){const r={};queueMicrotask(()=>{disk.set(k,structuredClone(v));tx.oncomplete?.();});return r;}}}};return tx;}};req.onsuccess();});return req;}};
const secrets=new Map();
globalThis.localStorage={setItem(k,v){assert.ok(['dsv4f.api-key','dsv4f.sidebar-width'].includes(k));secrets.set(k,v);},getItem(k){return secrets.get(k)??null;}};
const windowEvents=new Map();
globalThis.location={hash:'#chat'};globalThis.history={replaceState(){}};globalThis.window={innerWidth:1200,addEventListener(name,fn){windowEvents.set(name,fn);}};
globalThis.requestAnimationFrame=fn=>queueMicrotask(fn);
globalThis.setTimeout=()=>1;globalThis.clearTimeout=()=>{};globalThis.setInterval=()=>1;globalThis.clearInterval=()=>{};
let requests=[], responseMode='complete';
globalThis.fetch=async (url,options={})=>{
 if(url==='/health')return Response.json({status:'ok'});
 if(url==='/props')assert.equal(options.headers.Authorization,`Bearer ${secrets.get('dsv4f.api-key')??'dummykey'}`);
 if(url==='/props')return Response.json({model_alias:'dsv4f',model_ftype:'MXFP4 MoE',total_slots:1,default_generation_settings:{n_ctx:65536,params:{'speculative.types':'none'}}});
 if(url==='./benchmark.json')return Response.json({date:'2026-10-12',hardware:'RX7900XTX',median:15.73,runs:[15.73],prompt:'Benchmark',conditions:'test'});
 if(url==='/v1/chat/completions'){
  assert.equal(options.headers.Authorization,`Bearer ${secrets.get('dsv4f.api-key')??'dummykey'}`);
  const body=JSON.parse(options.body);requests.push(body);
  assert.equal(body.api_key,undefined);
  if(responseMode==='error')return Response.json({error:{message:'Test context exceeded'}},{status:400});
  if(responseMode==='stop')return new Response(new ReadableStream({start(stream){stream.enqueue(new TextEncoder().encode('data: '+JSON.stringify({choices:[{delta:{content:'Partial reply'}}],timings:{predicted_n:3,predicted_per_second:10}})+'\n\n'));options.signal.addEventListener('abort',()=>stream.error(new DOMException('Stopped','AbortError')));}}));
  const chunks=[{choices:[{delta:{reasoning_content:body.reasoning_effort==='high'?'Thinking briefly.':''}}],timings:{predicted_n:2,predicted_per_second:15.5}}, {choices:[{delta:{content:'Hello!'},finish_reason:'stop'}],timings:{predicted_n:6,predicted_per_second:16.25},usage:{completion_tokens:6,prompt_tokens:10}}];
  return new Response(chunks.map(c=>'data: '+JSON.stringify(c)+'\r\n\r\n').join('')+'data: [DONE]\n\n');
 }
 throw Error(url);
};
function dom(legacy=false){const els=new Map();for(const m of html.matchAll(/<([\w-]+)[^>]*\bid="([^"]+)"[^>]*>/g))els.set(m[2],new Element(m[1]));if(legacy)els.delete('sidebar-resizer');globalThis.document={documentElement:new Element('html'),scrollingElement:{scrollTop:0,scrollHeight:2000},getElementById:id=>els.get(id),body:new Element('body'),createElement:t=>new Element(t),createTextNode:t=>{const e=new Element('#text');e.textContent=t;return e;},addEventListener(){}};return id=>els.get(id);}
const flush=async()=>{for(let i=0;i<30;i++)await Promise.resolve();};
let $=dom();await import(new URL('../web/app.js?smoke=1', import.meta.url));await flush();
assert.equal($('chat-count').textContent,'1');assert.equal($('send').disabled,false);
const handle=$('sidebar-resizer');
handle.onpointerdown({button:0,isPrimary:true,pointerId:1,clientX:260,preventDefault(){}});
handle.onpointermove({pointerId:2,clientX:500});assert.equal(document.documentElement.style['--sidebar-width'],'260px');
handle.onpointermove({pointerId:1,clientX:340});assert.equal(document.documentElement.style['--sidebar-width'],'340px');
handle.onpointerup({pointerId:1});assert.equal(secrets.get('dsv4f.sidebar-width'),'340');assert.equal(document.body.classList.contains('sidebar-resizing'),false);
handle.onkeydown({key:'End',preventDefault(){}});assert.equal(document.documentElement.style['--sidebar-width'],'480px');
handle.onkeydown({key:'ArrowLeft',preventDefault(){}});assert.equal(document.documentElement.style['--sidebar-width'],'464px');
window.innerWidth=800;windowEvents.get('resize')();assert.equal(document.documentElement.style['--sidebar-width'],'440px');
window.innerWidth=1200;windowEvents.get('resize')();assert.equal(document.documentElement.style['--sidebar-width'],'464px');
assert.equal($('chat-list').children[0].querySelector('.history-delete').textContent,'×');
$('prompt').value='Hello test';await $('form').onsubmit({preventDefault(){}});await flush();
assert.equal(requests[0].reasoning_effort,'none');assert.equal($('messages').children.length,2);assert.match($('messages').textContent,/16.25 tok\/s/);
assert.equal(document.scrollingElement.scrollTop,document.scrollingElement.scrollHeight);
assert.match($('report-rows').textContent,/Hello test/);assert.equal(disk.get('dsv4f.chats.v1').chats[0].messages[1].state,'complete');
$('new-chat').onclick();await flush();assert.equal($('chat-count').textContent,'2');
$('search').value='Hello test';$('search').oninput();assert.equal($('chat-list').children.length,1);$('chat-list').children[0].querySelector('.history-item').onclick();await flush();assert.match($('messages').textContent,/Hello!/);
$('rename-chat').onclick();$('chat-title').value='Renamed chat';$('chat-dialog').returnValue='confirm';$('chat-dialog').close();await flush();assert.equal($('page-title').textContent,'Renamed chat');
$('delete-chat').onclick();$('chat-dialog').close();assert.equal($('chat-count').textContent,'2'); // Escape/cancel must not reuse a previous confirmation.
$('settings-open').onclick();$('s-api-key').value='test-custom-key';$('s-temperature').value='1.25';$('s-top-p').value='0.8';$('s-top-k').value='12';$('s-reasoning').value='high';await $('settings-form').onsubmit({preventDefault(){}});$('settings-close').onclick();await flush();
assert.equal(disk.get('dsv4f.chats.v1').settings.temperature,1.25);assert.equal(secrets.get('dsv4f.api-key'),'test-custom-key');assert.ok(!JSON.stringify(disk.get('dsv4f.chats.v1')).includes('test-custom-key'));
$('prompt').value='Second test';await $('form').onsubmit({preventDefault(){}});await flush();assert.equal(requests[1].temperature,1.25);assert.equal(requests[1].top_p,.8);assert.equal(requests[1].top_k,12);assert.equal(requests[1].reasoning_effort,'high');assert.equal($('messages').lastElementChild.querySelector('.reasoning').hidden,false);
// Simulate a fresh document opening against the saved IndexedDB state.
$=dom();await import(new URL('../web/app.js?smoke=2', import.meta.url));await flush();
assert.equal(document.documentElement.style['--sidebar-width'],'464px');
assert.equal($('chat-count').textContent,'2');assert.equal($('page-title').textContent,'Renamed chat');assert.equal($('s-reasoning').value,'high');assert.equal($('s-api-key').value,'test-custom-key');assert.match($('messages').textContent,/Thinking briefly/);assert.match($('messages').textContent,/16.25 tok\/s/);
responseMode='stop';$('prompt').value='Stop test';const pending=$('form').onsubmit({preventDefault(){}});await flush();
for(const row of $('chat-list').children){assert.equal(row.querySelector('.history-item').disabled,true);assert.equal(row.querySelector('.history-delete').disabled,true);row.querySelector('.history-delete').onclick();assert.equal(Boolean($('chat-dialog').open),false);}
const root=document.scrollingElement;
assert.equal(root.scrollTop,root.scrollHeight); // New output follows the bottom.
root.scrollTop=0;root.scrollHeight=3000;windowEvents.get('scroll')();await flush();assert.equal(root.scrollTop,3000);
const reasoning=$('messages').lastElementChild.querySelector('.reasoning-body');reasoning.scrollHeight=600;reasoning.scrollTop=0;
windowEvents.get('resize')();await flush();assert.equal(reasoning.scrollTop,600);
$('tab-report').onclick();root.scrollTop=0;windowEvents.get('scroll')();await flush();assert.equal(root.scrollTop,0); // Reports remain readable while generating.
$('tab-chat').onclick();await flush();assert.equal(root.scrollTop,3000);
root.scrollHeight=4000;windowEvents.get('resize')();await flush();assert.equal(root.scrollTop,4000);
$('stop').onclick();await pending;await flush();assert.equal(disk.get('dsv4f.chats.v1').chats.find(c=>c.title==='Renamed chat').messages.at(-1).state,'stopped');assert.match($('messages').textContent,/Partial reply/);
root.scrollTop=0;windowEvents.get('scroll')();await flush();assert.equal(root.scrollTop,0); // Finished responses permit normal scrolling.
responseMode='error';$('prompt').value='Error test';await $('form').onsubmit({preventDefault(){}});await flush();assert.equal($('error').textContent,'Test context exceeded');assert.equal(disk.get('dsv4f.chats.v1').chats.find(c=>c.title==='Renamed chat').messages.at(-1).state,'error');
// Delete another chat directly from the sidebar without switching or losing the active draft.
$('search').value='';$('search').oninput();$('prompt').value='Keep unsent draft';
let other=$('chat-list').children.find(row=>!row.querySelector('.history-item').attributes['aria-current']);
other.querySelector('.history-delete').onclick();assert.match($('dialog-description').textContent,/New chat/);assert.equal($('page-title').textContent,'Renamed chat');
$('chat-dialog').close();await flush();assert.equal($('chat-count').textContent,'2'); // Cancel must keep both chats.
other.querySelector('.history-delete').onclick();$('chat-dialog').returnValue='confirm';$('chat-dialog').close();await flush();
assert.equal($('chat-count').textContent,'1');assert.equal($('page-title').textContent,'Renamed chat');assert.equal($('prompt').value,'Keep unsent draft');assert.match($('messages').textContent,/Partial reply/);
assert.equal(disk.get('dsv4f.chats.v1').chats[0].title,'Renamed chat');assert.equal(disk.get('dsv4f.chats.v1').chats[0].draft,'Keep unsent draft');assert.equal($('report-rows').children.length,4);
// Deleting the final active chat leaves a usable empty conversation, including after reload.
$('chat-list').children[0].querySelector('.history-delete').onclick();$('chat-dialog').returnValue='confirm';$('chat-dialog').close();await flush();assert.equal($('chat-count').textContent,'1');assert.equal(disk.get('dsv4f.chats.v1').chats.length,1);assert.equal($('page-title').textContent,'New chat');assert.equal($('prompt').value,'');
$=dom(true);await import(new URL('../web/app.js?smoke=3', import.meta.url));await flush();assert.equal($('chat-count').textContent,'1');assert.equal($('page-title').textContent,'New chat');assert.ok(!JSON.stringify(disk.get('dsv4f.chats.v1')).includes('Renamed chat'));
assert.equal($('sidebar').lastElementChild.id,'sidebar-resizer');assert.equal(typeof $('sidebar').lastElementChild.onpointerdown,'function');
console.log('PASS: sidebar drag/keyboard resize, width persistence and cached-HTML compatibility, chat creation, streaming metrics and bottom scrolling, history search/switch/rename/targeted sidebar deletion/cancel, active draft preservation and last-chat deletion, saved settings in requests, reasoning display, reload persistence, partial stop and API error recovery (DOM/IndexedDB doubles).');
