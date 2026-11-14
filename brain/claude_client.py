"""Claude API backend for test 1 (reports/Notebook reading at shipping size 2026.md, amendment 2, section 8).

Gives the test scripts the same small interface as order_gap.Endpoint (request, chat, propose_batch, map), so
brain/notebook_read.py, brain/scientist.py and brain/longmem.py run unchanged on the Claude API:
  - official Anthropic SDK; the key is read from NICHTEC_CLAUDE_KEY (never ANTHROPIC_API_KEY, which would make the
    Claude Code session itself run on the owner's key);
  - thinking on = adaptive thinking at the model's default effort; thinking off = {"type": "disabled"} (Claude Haiku
    5.5 accepts it at its default effort). No temperature, top_p, top_k or seed: current Claude models reject them;
  - the notes part of a prompt (everything before the last "Question: ") is sent as its own cached block, so a
    5,000-note notebook re-read for every question is billed at the cache-read price; the question follows uncached;
  - a shared budget: every reply's reported usage is priced and added to results/claude_spend.json; no new request
    starts once the total reaches the cap (default $25).
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import anthropic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEND_FILE = os.path.join(ROOT, "results", "claude_spend.json")
# $ per million tokens (Anthropic model table, 2026-11-03). Haiku 5.5's price rises for prompts over 100K tokens.
PRICES = {"claude-haiku-5-5": {"in": 0.10, "out": 0.50, "long_in": 0.50, "long_out": 2.50, "long_from": 100_000,
                               "read": 0.1, "write": 1.25}}
QUESTION = "Question: "


class Budget:
    """Running total of estimated spend, shared by every script through one file; stops new requests at the cap."""

    def __init__(self, cap=25.0, path=SPEND_FILE):
        self.cap, self.path, self.lock = cap, path, threading.Lock()
        self.spent = json.load(open(path))["usd"] if os.path.exists(path) else 0.0

    def check(self):
        if self.spent >= self.cap:
            raise SystemExit(f"Claude budget reached: ${self.spent:.2f} of ${self.cap:.2f}; no new requests")

    def add(self, model, u):
        p = PRICES[model]
        prompt = u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)
        long = prompt > p["long_from"]
        pin, pout = (p["long_in"], p["long_out"]) if long else (p["in"], p["out"])
        usd = (u.input_tokens * pin + (u.cache_creation_input_tokens or 0) * pin * p["write"] +
               (u.cache_read_input_tokens or 0) * pin * p["read"] + u.output_tokens * pout) / 1e6
        with self.lock:
            self.spent += usd
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"usd": self.spent, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
            os.replace(tmp, self.path)
        return usd


class ClaudeEndpoint:
    def __init__(self, model="claude-haiku-5-5", budget=None, workers=16, cache=True):
        if model not in PRICES:
            raise SystemExit(f"{model}: no price on file; add it to PRICES before running")
        key = os.environ.get("NICHTEC_CLAUDE_KEY")
        if not key:
            raise SystemExit("NICHTEC_CLAUDE_KEY is not set (add it in the environment settings as a variable)")
        self.client = anthropic.Anthropic(api_key=key, base_url="https://api.anthropic.com", max_retries=8)
        self.model, self.served, self.workers = model, [model], workers
        self.budget = budget or Budget()
        self.lock, self.refusals, self.cache = threading.Lock(), 0, cache

    def blocks(self, content):
        """The notes as one cached block, the question as a second block (identical text, cheaper re-reads). Off for
        prompts read only once (each LongMemEval history), where a cache write would only add its 25% fee."""
        cut = content.rfind(QUESTION)
        if cut <= 0 or not self.cache:
            return [{"type": "text", "text": content}]
        return [{"type": "text", "text": content[:cut], "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": content[cut:]}]

    def create(self, content, thinking, max_tokens, raw=False):
        self.budget.check()
        kw = {"model": self.model, "max_tokens": max_tokens,
              "thinking": {"type": "adaptive"} if thinking else {"type": "disabled"},
              "messages": [{"role": "user", "content": self.blocks(content)}]}
        for attempt in range(60):                 # the SDK already retried 8 times; keep waiting out the account's
            try:                                  # rate limit (resumable run) rather than crash
                if raw:
                    resp = self.client.messages.with_raw_response.create(**kw)
                    msg = resp.parse()
                    self.budget.add(self.model, msg.usage)
                    return msg, {k: v for k, v in resp.headers.items() if k.startswith("anthropic-ratelimit")}
                msg = self.client.messages.create(**kw)
                self.budget.add(self.model, msg.usage)
                return msg
            except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError):
                time.sleep(min(60, 5 * (attempt + 1)))
                self.budget.check()
            except anthropic.BadRequestError as e:   # e.g. a prompt longer than the context: report, don't retry
                raise SystemExit(f"Claude API 400: {e.message[:300]}")
        raise SystemExit("Claude API: still failing after 60 waits")

    def request(self, method, path, body):
        """order_gap.Endpoint's interface: an OpenAI-shaped body in, an OpenAI-shaped reply out."""
        content = body["messages"][0]["content"]
        thinking = bool((body.get("chat_template_kwargs") or {}).get("enable_thinking"))
        msg = self.create(content, thinking, body.get("max_tokens", 1024))
        text = "".join(b.text for b in msg.content if b.type == "text")
        finish = {"max_tokens": "length", "refusal": "refusal"}.get(msg.stop_reason, "stop")
        if msg.stop_reason == "refusal":
            with self.lock:
                self.refusals += 1
            text = ""
        u = msg.usage
        cached = u.cache_read_input_tokens or 0
        return {"choices": [{"message": {"content": text}, "finish_reason": finish}],
                "usage": {"prompt_tokens": u.input_tokens + cached + (u.cache_creation_input_tokens or 0),
                          "completion_tokens": u.output_tokens, "prompt_tokens_details": {"cached_tokens": cached}}}

    def chat(self, content, thinking=False, **kw):
        return self.request("POST", "/chat/completions", {"messages": [{"role": "user", "content": content}],
                                                          "chat_template_kwargs": {"enable_thinking": thinking},
                                                          "max_tokens": kw.get("max_tokens", 1024)})

    def propose_batch(self, prompts, n, seeds):
        """The scientist agent's guesses: n separate calls per prompt (the API has no n). Thinking and the token cap
        follow order_gap.THINKING and order_gap.SAMPLING, as for the Qwen endpoints."""
        import order_gap as G
        out = []
        for p in prompts:
            replies = []
            for _ in range(n):
                r = self.chat(p, thinking=G.THINKING, max_tokens=G.SAMPLING["max_new_tokens"])
                ch = r["choices"][0]
                replies.append("" if ch["finish_reason"] != "stop" else ch["message"]["content"])
            out.append(replies)
        return out

    def map(self, f, xs):
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(f, xs))
