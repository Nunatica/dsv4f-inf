#!/usr/bin/env python3
"""Sequential reasoning-off benchmark using native prefill/decode timings."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from urllib.request import Request, urlopen

def api_headers():
    key = os.environ.get('API_KEY', 'dummykey')
    return {'Content-Type': 'application/json', **({'Authorization': 'Bearer '+key} if key else {})}


PROMPT = 'Explain story of the romance of three kingdoms.'


def generate(url, tokens, prompt=PROMPT):
    body = {'model': 'dsv4f', 'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': tokens, 'temperature': 0, 'seed': 42, 'cache_prompt': False,
            'chat_template_kwargs': {'enable_thinking': False}, 'reasoning_budget': 0,
            'timings_per_token': True}
    request = Request(url+'/v1/chat/completions', data=json.dumps(body).encode(), headers=api_headers())
    start = time.monotonic()
    with urlopen(request, timeout=1800) as response:
        result = json.load(response)
    elapsed = time.monotonic()-start
    msg = result['choices'][0]['message']
    if msg.get('reasoning_content'):
        raise RuntimeError('Benchmark generated reasoning despite reasoning-off settings')
    timing = result.get('timings')
    if not timing or not timing.get('prompt_per_second') or (tokens > 1 and not timing.get('predicted_per_second')):
        raise RuntimeError('Native server timings are absent')
    return {'wall_seconds': elapsed, 'timings': timing, 'usage': result.get('usage'),
            'finish_reason': result['choices'][0].get('finish_reason'), 'content': msg.get('content'),
            'reasoning_content': msg.get('reasoning_content'), 'response': result}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8080')
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--prompt-file', type=Path, help='UTF-8 prompt, including long prefill tests')
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--warmup-tokens', type=int, default=32)
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'logs'/'benchmark.json')
    args = p.parse_args()
    if args.tokens < 1 or args.runs < 1:
        p.error('tokens >= 1 and runs >= 1 are required')
    prompt = args.prompt_file.read_text(encoding='utf-8') if args.prompt_file else PROMPT
    with urlopen(Request(args.url+'/props', headers=api_headers()), timeout=10) as response:
        props = json.load(response)
    # Keep the exact owned-server configuration with the measurement. A foreign
    # server (for example the reference on 8066) must not inherit our manifest.
    engine_state = None
    state_path = Path(__file__).resolve().parents[1]/'state'/'server.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get('url') == args.url.rstrip('/'):
            engine_state = {k: state[k] for k in ('pid', 'command', 'engine_options', 'memory', 'load_seconds', 'log')}
            command = list(engine_state['command'])
            for i, arg in enumerate(command):
                if arg == '--api-key' and i + 1 < len(command):
                    command[i + 1] = '<redacted>'
            engine_state['command'] = command
    results = []
    if args.warmup_tokens:
        print('Warm-up...', flush=True)
        generate(args.url, args.warmup_tokens, prompt)
    for i in range(args.runs):
        result = generate(args.url, args.tokens, prompt)
        results.append(result)
        print(f"run {i+1}: prefill {result['timings']['prompt_per_second']:.2f} tok/s ({result['timings']['prompt_ms']:.2f} ms); decode {result['timings']['predicted_per_second']:.2f} tok/s; {result['timings']['predicted_n']} tokens", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = {'date_utc': datetime.now(timezone.utc).isoformat(), 'prompt': prompt,
                  'prompt_file': str(args.prompt_file) if args.prompt_file else None,
                  'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                  'engine_state': engine_state, 'max_tokens': args.tokens,
                  'cache_prompt': False, 'temperature': 0, 'seed': 42,
                  'reasoning': False, 'warmup_tokens': args.warmup_tokens, 'props': props,
                  'median_decode_tokens_per_second': statistics.median(r['timings']['predicted_per_second'] for r in results),
                  'median_prefill_tokens_per_second': statistics.median(r['timings']['prompt_per_second'] for r in results),
                  'runs': results}
        args.output.write_text(json.dumps(report, indent=2))
    print(f"Median prefill: {report['median_prefill_tokens_per_second']:.2f} tok/s; decode: {report['median_decode_tokens_per_second']:.2f} tok/s; {args.output}")


if __name__ == '__main__':
    main()
