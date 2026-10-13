import test from 'node:test';
import assert from 'node:assert/strict';
import {decodeRate, outputTokens, summarize, SSEParser} from '../web/metrics.js';
import {DEFAULTS, normalizeSettings, requestSettings} from '../web/settings.js';

test('native decode rate is never inferred from text or SSE event count', () => {
  assert.equal(decodeRate({content: 'a long response', usage: {completion_tokens: 100}}), null);
  assert.equal(decodeRate({timings: {predicted_n: 1, predicted_per_second: 999}}), null);
  assert.equal(decodeRate({timings: {predicted_n: 100, predicted_per_second: 15.73}}), 15.73);
  assert.equal(outputTokens({usage: {completion_tokens: 101}, timings: {predicted_n: 100}}), 101);
});
test('summary excludes cancelled and failed generation, and handles missing measurements', () => {
  const msg = (state, n, rate) => ({state, timings: {predicted_n: n, predicted_per_second: rate}});
  const s = summarize([msg('complete', 10, 12), msg('complete', 20, 18), msg('stopped', 30, 100), msg('error', 40, 150)]);
  assert.deepEqual(s, {responses: 2, tokens: 30, medianRate: 15, bestRate: 18, medianTtft: null});
  assert.equal(summarize([]).medianRate, null);
});
test('SSE parser survives every byte boundary, CRLF, unicode, comments and trailing event', () => {
  const wire = ': ping\r\ndata: {"choices":[{"delta":{"content":"안녕🌱"}}]}\r\n\r\ndata: {"usage":{"completion_tokens":2}}\n\ndata: [DONE]';
  const bytes = new TextEncoder().encode(wire);
  for (let split = 0; split <= bytes.length; split++) {
    const events = [], p = new SSEParser(data => events.push(data)), d = new TextDecoder();
    p.push(d.decode(bytes.slice(0, split), {stream: true})); p.push(d.decode(bytes.slice(split)), true);
    assert.equal(JSON.parse(events[0]).choices[0].delta.content, '안녕🌱');
    assert.equal(JSON.parse(events[1]).usage.completion_tokens, 2); assert.equal(events[2], '[DONE]'); assert.equal(events.length, 3);
  }
});
test('request applies all settings and overrides server default reasoning off/budget zero', () => {
  const off = requestSettings(DEFAULTS);
  assert.equal(off.reasoning_effort, 'none'); assert.equal(off.reasoning_budget_tokens, 0); assert.equal(off.chat_template_kwargs.enable_thinking, false);
  const on = requestSettings({temperature: 1.25, top_p: .8, top_k: 12, min_p: .02, max_tokens: 500, reasoning: 'high', reasoning_budget: 128});
  assert.equal(on.temperature, 1.25); assert.equal(on.top_p, .8); assert.equal(on.top_k, 12); assert.equal(on.min_p, .02); assert.equal(on.max_tokens, 500);
  assert.equal(on.reasoning_effort, 'high'); assert.equal(on.reasoning_format, 'deepseek'); assert.equal(on.reasoning_budget_tokens, 128); assert.equal(on.chat_template_kwargs.thinking, true);
});
test('corrupt or out-of-range stored settings fall back to valid defaults', () => {
  assert.deepEqual(normalizeSettings({temperature: -10, top_p: 4, top_k: 1.5, max_tokens: Infinity, reasoning: '<script>', reasoning_budget: 0}), DEFAULTS);
});
