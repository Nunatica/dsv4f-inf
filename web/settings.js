export const DEFAULTS = Object.freeze({temperature: 0.6, reasoning: 'none', reasoning_budget: 1024, top_p: 1, top_k: 40, min_p: 0.05, max_tokens: 2048});
export function normalizeSettings(raw = {}) {
  const out = {...DEFAULTS};
  for (const [key, min, max, integer] of [['temperature', 0, 2], ['top_p', .01, 1], ['top_k', 0, 100000, true], ['min_p', 0, 1], ['max_tokens', 1, 65536, true], ['reasoning_budget', 1, 65536, true]]) {
    const value = raw[key];
    if (typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max && (!integer || Number.isInteger(value))) out[key] = value;
  }
  if (['none', 'low', 'high', 'max'].includes(raw.reasoning)) out.reasoning = raw.reasoning;
  return out;
}
export function requestSettings(settings) {
  const s = normalizeSettings(settings), thinking = s.reasoning !== 'none';
  return {temperature: s.temperature, top_p: s.top_p, top_k: s.top_k, min_p: s.min_p, max_tokens: s.max_tokens,
    reasoning_effort: s.reasoning, reasoning_format: 'deepseek',
    chat_template_kwargs: {enable_thinking: thinking, thinking, reasoning_effort: s.reasoning},
    reasoning_budget_tokens: thinking ? s.reasoning_budget : 0};
}
