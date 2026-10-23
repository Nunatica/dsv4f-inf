import {decodeRate, outputTokens, summarize, SSEParser} from './metrics.js';
import {DEFAULTS, normalizeSettings, requestSettings} from './settings.js';

const $ = id => document.getElementById(id);
const KEY = 'dsv4f.chats.v1';
const number = (n, digits = 2) => Number.isFinite(n) ? n.toLocaleString('en-US', {maximumFractionDigits: digits, minimumFractionDigits: digits}) : '—';
const seconds = ms => Number.isFinite(ms) ? `${number(ms / 1000, 2)} s` : '—';
const date = iso => new Date(iso).toLocaleString('en-US', {month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'});
const make = (tag, className, text) => { const el = document.createElement(tag); if (className) el.className = className; if (text !== undefined) el.textContent = text; return el; };
const textOf = content => typeof content === 'string' ? content : (content || []).filter(x => x.type === 'text').map(x => x.text).join('\n');
const uid = () => {
  if (typeof crypto.randomUUID === 'function') return crypto.randomUUID();
  // LAN HTTP is not a secure context; getRandomValues is still available there.
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
};
const newChat = () => ({id: uid(), title: 'New chat', createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(), messages: [], draft: ''});
let chats = [], activeId, db, busy = false, controller = null, ready = false, props = null, benchmark = null, saveTimer, saveChain = Promise.resolve();
let view = location.hash === '#report' ? 'report' : 'chat';
let lastSpeed = 'Idle';
let settings = {...DEFAULTS};
let apiKey = 'dummykey';
try { apiKey = localStorage.getItem('dsv4f.api-key') ?? apiKey; } catch { /* Use the launcher default. */ }
const apiHeaders = () => apiKey ? {Authorization: `Bearer ${apiKey}`} : {};
const current = () => chats.find(c => c.id === activeId);
const SIDEBAR_WIDTH_KEY = 'dsv4f.sidebar-width';
let sidebarWidth = 260, sidebarDrag = null;
// Keep fresh JS compatible with an older cached HTML document.
const sidebarResizer = $('sidebar-resizer') || make('div');
sidebarResizer.id = 'sidebar-resizer';
for (const [name, value] of Object.entries({role: 'separator', tabindex: '0', 'aria-label': 'Resize sidebar', 'aria-orientation': 'vertical', 'aria-controls': 'sidebar', title: 'Drag to resize the sidebar · Arrow keys also work'})) sidebarResizer.setAttribute(name, value);
if (!$('sidebar-resizer')) $('sidebar').append(sidebarResizer);
try {
  const savedWidth = Number(localStorage.getItem(SIDEBAR_WIDTH_KEY));
  if (Number.isFinite(savedWidth) && savedWidth > 0) sidebarWidth = savedWidth;
} catch { /* Keep the default width when storage is unavailable. */ }
function sidebarBounds() {
  const max = Math.max(160, Math.min(480, window.innerWidth - (window.innerWidth <= 760 ? 48 : 360)));
  return {min: Math.min(220, max), max};
}
function applySidebarWidth(value = sidebarWidth) {
  const {min, max} = sidebarBounds(), width = Math.max(min, Math.min(max, Math.round(value)));
  document.documentElement.style.setProperty('--sidebar-width', `${width}px`);
  const handle = sidebarResizer;
  handle.setAttribute('aria-valuemin', String(min)); handle.setAttribute('aria-valuemax', String(max));
  handle.setAttribute('aria-valuenow', String(width)); handle.setAttribute('aria-valuetext', `${width} pixels`);
  return width;
}
function saveSidebarWidth() {
  try { localStorage.setItem(SIDEBAR_WIDTH_KEY, String(sidebarWidth)); } catch { /* Resizing still works without persistence. */ }
}
sidebarResizer.onpointerdown = event => {
  if (event.button !== 0 || !event.isPrimary) return;
  event.preventDefault();
  sidebarDrag = {id: event.pointerId, x: event.clientX, width: applySidebarWidth()};
  sidebarResizer.setPointerCapture(event.pointerId);
  document.body.classList.toggle('sidebar-resizing', true);
};
sidebarResizer.onpointermove = event => {
  if (!sidebarDrag || sidebarDrag.id !== event.pointerId) return;
  sidebarWidth = applySidebarWidth(sidebarDrag.width + event.clientX - sidebarDrag.x);
};
function finishSidebarResize(event) {
  if (!sidebarDrag || sidebarDrag.id !== event.pointerId) return;
  sidebarDrag = null;
  document.body.classList.toggle('sidebar-resizing', false);
  if (sidebarResizer.hasPointerCapture(event.pointerId)) sidebarResizer.releasePointerCapture(event.pointerId);
  saveSidebarWidth();
}
sidebarResizer.onpointerup = sidebarResizer.onpointercancel = sidebarResizer.onlostpointercapture = finishSidebarResize;
sidebarResizer.onkeydown = event => {
  const {min, max} = sidebarBounds(), width = applySidebarWidth();
  const next = {ArrowLeft: width - 16, ArrowRight: width + 16, Home: min, End: max}[event.key];
  if (next === undefined) return;
  event.preventDefault(); sidebarWidth = applySidebarWidth(next); saveSidebarWidth();
};
applySidebarWidth();
let bottomScrollPending = false;
function followOutput(force = false) {
  if (view !== 'chat' || (!busy && !force) || bottomScrollPending) return;
  bottomScrollPending = true;
  // Coalesce token updates and scroll events into one scroll after layout.
  requestAnimationFrame(() => {
    bottomScrollPending = false;
    if (view !== 'chat') return;
    const root = document.scrollingElement;
    if (root) root.scrollTop = root.scrollHeight;
    const reasoning = $('messages').lastElementChild?.querySelector('.reasoning-body');
    if (reasoning) reasoning.scrollTop = reasoning.scrollHeight;
  });
}
function storageWarning(text) { $('storage-warning').textContent = text; $('storage-warning').hidden = false; }
function dbRequest(mode, action) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction('state', mode), request = action(tx.objectStore('state'));
    tx.oncomplete = () => resolve(request.result);
    tx.onerror = tx.onabort = () => reject(tx.error || request.error || Error('Failed to save'));
  });
}
async function load() {
  let saved;
  try {
    db = await new Promise((resolve, reject) => {
      const req = indexedDB.open('dsv4f-chat', 1);
      req.onupgradeneeded = () => req.result.createObjectStore('state');
      req.onsuccess = () => resolve(req.result); req.onerror = () => reject(req.error);
      req.onblocked = () => reject(Error('Storage is in use by another tab.'));
    });
    saved = await dbRequest('readonly', store => store.get(KEY));
  } catch {
    db = null;
    storageWarning('The browser database is unavailable. Using fallback storage, which may run out of space if you save many images.');
    try { saved = JSON.parse(localStorage.getItem(KEY) || 'null'); } catch { /* Keep an empty in-memory state. */ }
  }
  if (saved?.version === 1 && Array.isArray(saved.chats)) {
    chats = saved.chats.filter(c => typeof c.id === 'string' && typeof c.title === 'string' && Array.isArray(c.messages));
    activeId = saved.activeId;
    settings = normalizeSettings(saved.settings);
    for (const chat of chats) for (const message of chat.messages) if (message.state === 'streaming') message.state = 'interrupted';
  }
  if (!chats.length) chats.push(newChat());
  if (!current()) activeId = chats[0].id;
}
function persist() {
  clearTimeout(saveTimer);
  saveTimer = null;
  // Snapshot before queuing so later edits cannot change an earlier transaction.
  const snapshot = structuredClone({version: 1, chats, activeId, settings});
  saveChain = saveChain.then(async () => {
    try {
      if (db) await dbRequest('readwrite', store => store.put(snapshot, KEY));
      else localStorage.setItem(KEY, JSON.stringify(snapshot));
    } catch {
      storageWarning('Could not save the conversation. Check available storage or browser permissions. The conversation is still on screen, but recent changes may be lost if you refresh.');
    }
  });
  return saveChain;
}
function scheduleSave() { if (!saveTimer) saveTimer = setTimeout(() => { saveTimer = null; persist(); }, 1200); }
function menu(open) { document.body.classList.toggle('menu-open', open); $('scrim').hidden = !open; $('menu-toggle').setAttribute('aria-expanded', String(open)); }
function setView(next) {
  view = next; $('view-chat').hidden = view !== 'chat'; $('view-report').hidden = view !== 'report';
  for (const name of ['chat', 'report']) {
    $(`tab-${name}`).classList.toggle('active', name === view);
    if (name === view) $(`tab-${name}`).setAttribute('aria-current', 'page'); else $(`tab-${name}`).removeAttribute('aria-current');
  }
  $('eyebrow').textContent = view === 'chat' ? 'Conversation' : 'Report';
  $('page-title').textContent = view === 'chat' ? current().title : 'Performance';
  history.replaceState(null, '', view === 'report' ? '#report' : '#chat');
  if (view === 'report') renderReport();
  menu(false);
  followOutput();
}
function renderHistory() {
  const query = $('search').value.trim().toLocaleLowerCase();
  $('chat-list').replaceChildren(); $('chat-count').textContent = chats.length;
  for (const chat of [...chats].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))) {
    if (query && !`${chat.title} ${chat.messages.map(m => textOf(m.content)).join(' ')}`.toLocaleLowerCase().includes(query)) continue;
    const row = make('div', `history-row${chat.id === activeId ? ' active' : ''}`);
    const button = make('button', 'history-item');
    button.type = 'button'; button.disabled = busy; button.title = chat.title;
    if (chat.id === activeId) button.setAttribute('aria-current', 'true');
    const messageCount = chat.messages.filter(m => m.role === 'user').length;
    button.append(make('strong', '', chat.title), make('small', '', `${date(chat.updatedAt)} · ${messageCount} ${messageCount === 1 ? 'message' : 'messages'}`));
    button.onclick = () => selectChat(chat.id);
    const remove = make('button', 'history-delete danger', '×');
    remove.type = 'button'; remove.disabled = busy;
    remove.title = `Delete “${chat.title}”`; remove.setAttribute('aria-label', remove.title);
    remove.onclick = () => openDialog('delete', chat.id);
    row.append(button, remove); $('chat-list').append(row);
  }
  if (!$('chat-list').childElementCount) $('chat-list').append(make('p', 'history-empty', 'No conversations found.'));
}
function selectChat(id) {
  if (busy) return;
  current().draft = $('prompt').value; activeId = id; $('error').textContent = ''; clearImage();
  renderChat(); setView('chat'); renderHistory(); persist();
  followOutput(true);
}
function messageNode(message) {
  const card = make('article', `message ${message.role}`);
  const label = make('div', 'message-label'); label.append(make('span', '', message.role === 'user' ? 'You' : 'DEEPSEEK FLASH'), make('span', '', date(message.createdAt)));
  const body = make('div', 'message-body', textOf(message.content)); card.append(label, body);
  if (Array.isArray(message.content)) for (const part of message.content) if (part.type === 'image_url' && /^data:image\//.test(part.image_url?.url)) {
    const image = make('img'); image.src = part.image_url.url; image.alt = 'Attached image'; card.append(image);
  }
  let metrics, error;
  if (message.role === 'assistant') {
    const reasoning = make('details', 'reasoning'); reasoning.hidden = !message.reasoning_content;
    reasoning.append(make('summary', '', 'Reasoning'), make('div', 'reasoning-body', message.reasoning_content || '')); card.insertBefore(reasoning, body);
    metrics = make('div', 'message-metrics'); error = make('div', 'message-error'); card.append(metrics, error);
  }
  $('messages').append(card);
  const update = () => {
    body.textContent = textOf(message.content) || (message.state === 'streaming' ? 'Waiting for a response…' : 'No response was generated.');
    if (!metrics) return;
    metrics.replaceChildren(make('span', 'speed', decodeRate(message) === null ? 'Speed —' : `${number(decodeRate(message))} tok/s`));
    const n = outputTokens(message), state = {streaming: 'Generating', complete: 'Complete', stopped: 'Stopped', interrupted: 'Disconnected', error: 'Failed'}[message.state] || '';
    metrics.append(document.createTextNode(` · ${n === null ? 'Awaiting token count' : `${number(n, 0)} tokens`} · ${state}${message.finishReason === 'length' ? ' (output limit reached)' : ''}${Number.isFinite(message.elapsedMs) ? ` · ${seconds(message.elapsedMs)}` : ''}`));
    error.textContent = message.error || '';
  };
  update(); return update;
}
function renderChat() {
  $('messages').replaceChildren(); $('prompt').value = current().draft || '';
  $('chat-date').textContent = `Started ${date(current().createdAt)}`;
  if (!current().messages.length) {
    const intro = make('div', 'empty-state'); intro.append(make('div', 'empty-mark', '✳'), make('h2', '', 'What would you like to know?'), make('p', '', 'Chat with your local model.\nConversations and generation speed are recorded together.')); $('messages').append(intro);
  } else for (const message of current().messages) messageNode(message);
  $('rename-chat').disabled = busy; $('delete-chat').disabled = busy;
  if (view === 'chat') $('page-title').textContent = current().title;
}
function rows() { return chats.flatMap(chat => chat.messages.filter(m => m.role === 'assistant').map(message => ({chat, message}))).sort((a, b) => b.message.createdAt.localeCompare(a.message.createdAt)); }
function renderReport() {
  const all = rows(), stats = summarize(all.map(r => r.message));
  $('report-cards').replaceChildren();
  for (const [label, value, unit] of [['Median decoding speed', number(stats.medianRate), 'tok/s'], ['Best decoding speed', number(stats.bestRate), 'tok/s'], ['Completed responses', number(stats.responses, 0), 'responses'], ['Total output tokens', number(stats.tokens, 0), 'tokens']]) {
    const card = make('div', 'metric-card'), v = make('div', 'metric-value', value); v.append(make('span', '', unit)); card.append(make('small', '', label), v); $('report-cards').append(card);
  }
  $('report-count').textContent = `${all.length} ${all.length === 1 ? 'record' : 'records'}`; $('report-rows').replaceChildren();
  for (const {chat, message: m} of all) {
    const tr = make('tr'), title = make('td'), button = make('button', 'table-chat', chat.title); button.type = 'button'; button.disabled = busy; button.onclick = () => selectChat(chat.id);
    title.append(button, make('small', '', `${date(m.createdAt)} · Reasoning: ${m.settings?.reasoning || 'none'}`)); tr.append(title);
    for (const value of [{streaming: 'Generating', complete: 'Complete', stopped: 'Stopped', error: 'Failed', interrupted: 'Disconnected'}[m.state] || '—', number(outputTokens(m), 0), decodeRate(m) === null ? '—' : `${number(decodeRate(m))} tok/s`, seconds(m.ttftMs), seconds(m.elapsedMs)]) tr.append(make('td', '', value));
    $('report-rows').append(tr);
  }
  if (!all.length) { const td = make('td', 'muted', 'No response measurements yet. Start a chat to record them here.'); td.colSpan = 6; const tr = make('tr'); tr.append(td); $('report-rows').append(tr); }
}
function renderSettings() {
  if (!props) return;
  const generation = props.default_generation_settings || {}, params = generation.params || {};
  const items = [['Model', props.model_alias || 'dsv4f'], ['Context capacity', number(generation.n_ctx, 0)], ['Quantization', props.model_ftype || '—'], ['Reasoning mode', settings.reasoning + ' (next request)'], ['Concurrent slots', number(props.total_slots, 0)], ['Speculative decoding', params['speculative.types'] || '—']];
  $('server-settings').replaceChildren();
  for (const [label, value] of items) { const cell = make('div'); cell.append(make('div', 'setting-label', label), make('div', 'setting-value', value)); $('server-settings').append(cell); }
}
async function loadBenchmark() {
  try {
    const r = await fetch('./benchmark.json', {cache: 'no-store'}); if (!r.ok) throw Error(); benchmark = await r.json();
    const el = $('benchmark'); el.replaceChildren();
    const summary = make('div', 'benchmark-summary', number(benchmark.median)); summary.append(make('small', '', 'tok/s · median'));
    const runs = make('div', 'benchmark-runs'); benchmark.runs.forEach((r, i) => runs.append(make('span', '', `Run ${i + 1} · ${number(r)} tok/s`)));
    el.append(make('p', '', `Measured on ${new Date(benchmark.date).toLocaleDateString('en-US')} · ${benchmark.hardware}`), summary, make('p', '', benchmark.prompt), runs, make('p', 'note', benchmark.conditions), make('p', 'note', 'This is a saved benchmark and does not represent live performance in the current chat.'));
  } catch { $('benchmark').textContent = 'No saved engine benchmark is available.'; }
}
function syncBusy() {
  for (const id of ['new-chat', 'rename-chat', 'delete-chat', 'image', 'prompt', 'remove-image']) $(id).disabled = busy;
  $('send').disabled = busy || !ready; $('stop').hidden = !controller; renderHistory();
}
async function health() {
  const wasReady = ready;
  try {
    const r = await fetch('/health', {signal: AbortSignal.timeout(4000)}); ready = r.ok;
    $('status').textContent = ready ? 'Model ready' : 'Loading model';
    if (ready && (!props || !wasReady)) {
      try { const p = await fetch('/props', {headers: apiHeaders(), signal: AbortSignal.timeout(4000)}); if (p.ok) { props = await p.json(); renderSettings(); } } catch { /* Retry on next poll. */ }
    }
  } catch { ready = false; $('status').textContent = 'Cannot connect to the server'; }
  $('status-dot').classList.toggle('ready', ready); $('send').disabled = !ready || busy;
  if (!busy) $('live-speed').textContent = ready ? lastSpeed : 'Waiting for server connection';
  setTimeout(health, ready ? 10000 : 3000);
}
function clearImage() { $('image').value = ''; $('attachment-name').textContent = ''; $('remove-image').hidden = true; }
function readImage(file) {
  if (file.size > 20 * 1024 * 1024) return Promise.reject(Error('Choose an image no larger than 20 MB.'));
  if (!file.type.startsWith('image/')) return Promise.reject(Error('Choose an image file.'));
  return new Promise((resolve, reject) => { const r = new FileReader(); r.onload = () => resolve(r.result); r.onerror = () => reject(Error('Could not read the image.')); r.readAsDataURL(file); });
}
$('form').onsubmit = async event => {
  event.preventDefault(); if (busy || !ready) return;
  const text = $('prompt').value.trim(); if (!text) return;
  busy = true; syncBusy(); $('error').textContent = '';
  const chat = current(); let content = text;
  try { const file = $('image').files[0]; if (file) content = [{type: 'text', text}, {type: 'image_url', image_url: {url: await readImage(file)}}]; }
  catch (err) { $('error').textContent = err.message; busy = false; syncBusy(); return; }
  const user = {role: 'user', content, createdAt: new Date().toISOString()};
  chat.messages.push(user); if (chat.title === 'New chat') chat.title = text.slice(0, 42);
  // Exclude empty failed assistant placeholders from the API conversation.
  const requestMessages = chat.messages.filter(m => m.role === 'user' || textOf(m.content)).map(({role, content, reasoning_content}) => ({role, content, ...(reasoning_content ? {reasoning_content} : {})}));
  const answer = {role: 'assistant', content: '', reasoning_content: '', settings: {...settings}, state: 'streaming', createdAt: new Date().toISOString()}; chat.messages.push(answer);
  chat.draft = ''; chat.updatedAt = answer.createdAt; $('prompt').value = ''; clearImage(); renderChat(); renderHistory();
  // Bind only the new card to stream updates; persisted content is the source of truth.
  const card = $('messages').lastElementChild;
  const updateMessage = () => {
    card.querySelector('.message-body').textContent = answer.content || (answer.state === 'streaming' ? 'Waiting for a response…' : 'No response was generated.');
    card.querySelector('.reasoning').hidden = !answer.reasoning_content;
    card.querySelector('.reasoning-body').textContent = answer.reasoning_content;
    const metrics = card.querySelector('.message-metrics'), n = outputTokens(answer), rate = decodeRate(answer);
    metrics.replaceChildren(make('span', 'speed', rate === null ? 'Speed —' : `${number(rate)} tok/s`), document.createTextNode(` · ${n === null ? 'Awaiting token count' : `${number(n, 0)} tokens`} · ${{streaming: 'Generating', complete: 'Complete', stopped: 'Stopped', interrupted: 'Disconnected', error: 'Failed'}[answer.state]}${answer.finishReason === 'length' ? ' (output limit reached)' : ''} · ${seconds(answer.elapsedMs)}`));
    card.querySelector('.message-error').textContent = answer.error || '';
    $('live-speed').textContent = rate === null ? (answer.content ? 'Generating · Awaiting speed measurement' : 'Waiting for the first token') : `${number(rate)} tok/s${busy ? ' · Generating' : ''}`;
    if (view === 'report') renderReport();
    followOutput(answer.state !== 'streaming');
  };
  controller = new AbortController(); syncBusy(); persist();
  const started = performance.now(); let reader, doneMarker = false;
  updateMessage();
  const timer = setInterval(() => { answer.elapsedMs = performance.now() - started; updateMessage(); }, 1000);
  try {
    const response = await fetch('/v1/chat/completions', {method: 'POST', headers: {'Content-Type': 'application/json', ...apiHeaders()}, signal: controller.signal,
      body: JSON.stringify({model: props?.model_alias || 'dsv4f', messages: requestMessages, stream: true, stream_options: {include_usage: true}, timings_per_token: true, ...requestSettings(answer.settings)})});
    if (!response.ok) { let detail = await response.text(); try { const j = JSON.parse(detail); detail = j.error?.message || detail; } catch {} throw Error(detail || `HTTP ${response.status}`); }
    if (!response.body) throw Error('Cannot read the streaming response.');
    reader = response.body.getReader(); const decoder = new TextDecoder();
    const parser = new SSEParser(data => {
      if (data === '[DONE]') { doneMarker = true; return; }
      const chunk = JSON.parse(data); if (chunk.error) throw Error(chunk.error.message || JSON.stringify(chunk.error));
      const delta = chunk.choices?.[0]?.delta?.content;
      if (typeof delta === 'string' && delta) { if (answer.ttftMs === undefined) answer.ttftMs = performance.now() - started; answer.content += delta; }
      const thinking = chunk.choices?.[0]?.delta?.reasoning_content;
      if (typeof thinking === 'string' && thinking) { if (answer.ttftMs === undefined) answer.ttftMs = performance.now() - started; answer.reasoning_content += thinking; }
      if (chunk.timings) answer.timings = chunk.timings;
      if (chunk.usage) answer.usage = chunk.usage;
      if (chunk.choices?.[0]?.finish_reason) answer.finishReason = chunk.choices[0].finish_reason;
      answer.elapsedMs = performance.now() - started; updateMessage(); scheduleSave();
    });
    while (!doneMarker) { const {value, done} = await reader.read(); parser.push(decoder.decode(value || new Uint8Array(), {stream: !done}), done); if (done) break; }
    if (!doneMarker) throw Error('The response stream was disconnected.');
    answer.state = 'complete';
  } catch (err) {
    answer.state = err.name === 'AbortError' ? 'stopped' : 'error';
    if (answer.state === 'error') { answer.error = err.message; $('error').textContent = err.message; }
  } finally {
    clearInterval(timer); if (reader) { try { await reader.cancel(); } catch {} reader.releaseLock(); }
    answer.elapsedMs = performance.now() - started; busy = false; controller = null;
    chat.updatedAt = new Date().toISOString(); updateMessage();
    lastSpeed = decodeRate(answer) === null ? 'Idle' : `${number(decodeRate(answer))} tok/s · Latest response`;
    $('live-speed').textContent = lastSpeed; syncBusy(); await persist(); renderReport(); $('prompt').focus();
  }
};
$('new-chat').onclick = () => { if (busy) return; current().draft = $('prompt').value; const c = newChat(); chats.push(c); activeId = c.id; $('search').value = ''; clearImage(); $('error').textContent = ''; renderChat(); renderHistory(); setView('chat'); persist(); $('prompt').focus(); };
$('stop').onclick = () => controller?.abort();
$('prompt').onkeydown = event => { if (event.ctrlKey && event.key === 'Enter') { event.preventDefault(); if (!busy && ready) $('form').requestSubmit(); } };
$('prompt').oninput = () => { current().draft = $('prompt').value; scheduleSave(); };
$('image').onchange = () => { $('attachment-name').textContent = $('image').files[0]?.name || ''; $('remove-image').hidden = !$('image').files.length; };
$('remove-image').onclick = clearImage;
$('search').oninput = renderHistory;
$('tab-chat').onclick = () => setView('chat'); $('tab-report').onclick = () => setView('report');
$('menu-toggle').onclick = () => menu(!document.body.classList.contains('menu-open')); $('scrim').onclick = () => menu(false);
window.addEventListener('keydown', event => { if (event.key === 'Escape' && !$('settings-panel').open && !$('chat-dialog').open) { menu(false); controller?.abort(); } });
window.addEventListener('hashchange', () => setView(location.hash === '#report' ? 'report' : 'chat'));
window.addEventListener('scroll', () => followOutput(), {passive: true});
window.addEventListener('resize', () => { applySidebarWidth(); followOutput(); });
document.addEventListener('visibilitychange', () => { if (document.hidden) persist(); });
let dialogAction, dialogChatId;
function openDialog(action, id = activeId) {
  if (busy) return;
  const chat = chats.find(c => c.id === id); if (!chat) return;
  dialogAction = action; dialogChatId = id;
  const deleting = action === 'delete'; $('dialog-title').textContent = deleting ? 'Delete conversation?' : 'Rename conversation';
  $('dialog-description').textContent = deleting ? `This will delete the messages and performance records for “${chat.title}” from this browser.` : 'Enter a name to display in the sidebar history.';
  $('title-label').hidden = $('chat-title').hidden = deleting; $('chat-title').required = !deleting; $('chat-title').value = chat.title;
  $('dialog-confirm').textContent = deleting ? 'Delete' : 'Save'; $('dialog-confirm').className = deleting ? 'danger' : 'primary';
  $('chat-dialog').returnValue = 'cancel'; $('chat-dialog').showModal();
}
$('rename-chat').onclick = () => openDialog('rename'); $('delete-chat').onclick = () => openDialog('delete');
$('dialog-form').onsubmit = event => { if (event.submitter?.value === 'confirm' && dialogAction === 'rename' && !$('chat-title').value.trim()) { event.preventDefault(); $('chat-title').focus(); } };
$('chat-dialog').onclose = () => {
  if ($('chat-dialog').returnValue !== 'confirm' || busy) return;
  const chat = chats.find(c => c.id === dialogChatId); if (!chat) return;
  current().draft = $('prompt').value;
  const affectsCurrent = chat.id === activeId;
  if (dialogAction === 'rename') { chat.title = $('chat-title').value.trim(); chat.updatedAt = new Date().toISOString(); }
  else {
    chats = chats.filter(c => c.id !== chat.id); if (!chats.length) chats.push(newChat());
    if (affectsCurrent) { activeId = chats[0].id; clearImage(); $('error').textContent = ''; }
  }
  if (affectsCurrent) renderChat();
  renderHistory(); renderReport(); persist();
};
$('export-report').onclick = () => {
  const report = {exportedAt: new Date().toISOString(), summary: summarize(rows().map(r => r.message)), server: props ? {model: props.model_alias, context: props.default_generation_settings?.n_ctx, quantization: props.model_ftype} : null,
    responses: rows().map(({chat, message: m}) => ({chatId: chat.id, title: chat.title, createdAt: m.createdAt, state: m.state, settings: m.settings, timings: m.timings, usage: m.usage, ttftMs: m.ttftMs, elapsedMs: m.elapsedMs})), benchmark};
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], {type: 'application/json'}));
  const link = make('a'); link.href = url; link.download = `deepseek-performance-${new Date().toISOString().slice(0, 10)}.json`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};
const settingIds = {temperature: 's-temperature', reasoning: 's-reasoning', reasoning_budget: 's-budget', top_p: 's-top-p', top_k: 's-top-k', min_p: 's-min-p', max_tokens: 's-max-tokens'};
function populateSettings(value) { $('s-api-key').value = apiKey; for (const [key, id] of Object.entries(settingIds)) $(id).value = value[key]; }
function closeSettings() { $('settings-panel').close(); $('settings-open').setAttribute('aria-expanded', 'false'); populateSettings(settings); }
$('settings-open').onclick = () => { populateSettings(settings); $('settings-feedback').hidden = true; $('settings-panel').showModal(); $('settings-open').setAttribute('aria-expanded', 'true'); };
$('settings-close').onclick = closeSettings;
$('settings-panel').addEventListener('click', event => { if (event.target === $('settings-panel')) { const r = event.target.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right) closeSettings(); } });
$('settings-panel').addEventListener('close', () => { $('settings-open').setAttribute('aria-expanded', 'false'); populateSettings(settings); });
$('settings-reset').onclick = () => { populateSettings(DEFAULTS); $('s-api-key').value = 'dummykey'; };
$('settings-form').onsubmit = async event => {
  event.preventDefault(); if (!$('settings-form').reportValidity()) return;
  settings = normalizeSettings(Object.fromEntries(Object.entries(settingIds).map(([key, id]) => [key, key === 'reasoning' ? $(id).value : Number($(id).value)])));
  apiKey = $('s-api-key').value.trim();
  try { localStorage.setItem('dsv4f.api-key', apiKey); } catch { storageWarning('Could not save the API key. Connection settings will remain active in this tab.'); }
  await persist(); renderSettings(); $('settings-feedback').hidden = false; $('settings-feedback').textContent = 'Saved. Changes apply starting with the next message.';
};
await load(); populateSettings(settings); renderChat(); renderHistory(); renderReport(); setView(view); persist(); health(); loadBenchmark();
