// Native counters only: SSE events and characters are not tokens.
export function decodeRate(message) {
  const t = message.timings;
  return t?.predicted_n > 1 && Number.isFinite(t.predicted_per_second) && t.predicted_per_second > 0 ? t.predicted_per_second : null;
}
export function outputTokens(message) {
  const n = message.usage?.completion_tokens ?? message.timings?.predicted_n;
  return Number.isFinite(n) && n >= 0 ? n : null;
}
export function median(values) {
  const a = values.filter(Number.isFinite).sort((a, b) => a - b), n = a.length;
  return n ? (n % 2 ? a[n >> 1] : (a[n / 2 - 1] + a[n / 2]) / 2) : null;
}
export function summarize(messages) {
  const complete = messages.filter(m => m.state === 'complete');
  return {responses: complete.length, tokens: complete.reduce((s, m) => s + (outputTokens(m) ?? 0), 0),
    medianRate: median(complete.map(decodeRate)), bestRate: complete.map(decodeRate).filter(Number.isFinite).reduce((a, b) => Math.max(a ?? b, b), null),
    medianTtft: median(complete.map(m => m.ttftMs).filter(Number.isFinite))};
}
export class SSEParser {
  constructor(onEvent) { this.buffer = ''; this.onEvent = onEvent; }
  push(text, final = false) {
    this.buffer += text;
    let at;
    while ((at = this.buffer.indexOf('\n')) >= 0) {
      const line = this.buffer.slice(0, at).replace(/\r$/, '');
      this.buffer = this.buffer.slice(at + 1); this.line(line);
    }
    if (final) { if (this.buffer) this.line(this.buffer.replace(/\r$/, '')); this.buffer = ''; this.line(''); }
  }
  line(line) {
    if (!line) { if (this.data?.length) this.onEvent(this.data.join('\n')); this.data = []; }
    else if (line.startsWith('data:')) (this.data ??= []).push(line.slice(5).replace(/^ /, ''));
  }
}
