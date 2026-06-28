#!/usr/bin/env python3
"""Test that triattention_page_budget option flows through the ollama API."""
import json
import sys
import time
import urllib.request

BASE = "http://localhost:11434"
MODEL = "qwen2.5-coder-test:latest"


def generate(options):
    req = urllib.request.Request(
        f"{BASE}/api/generate",
        data=json.dumps({
            "model": MODEL,
            "prompt": "Hello, say one word.",
            "stream": False,
            "options": options,
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def test(name, options):
    t = time.monotonic()
    try:
        resp = generate(options)
        elapsed = time.monotonic() - t
        if resp.get("response"):
            print(f"PASS [{name}] {elapsed:.1f}s: {resp['response'][:60]!r}")
            return True
        print(f"FAIL [{name}] empty response: {resp}")
        return False
    except Exception as e:
        print(f"FAIL [{name}] {e}")
        return False


results = [
    test("budget=512 (explicit)", {"triattention_page_budget": 512}),
    test("budget=0 (disabled)", {"triattention_page_budget": 0}),
    test("budget=-1 (auto)", {"triattention_page_budget": -1}),
]
print(f"\n{'ALL PASS' if all(results) else 'SOME FAILED'} ({sum(results)}/{len(results)})")
sys.exit(0 if all(results) else 1)
