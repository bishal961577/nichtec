"""Part C of test 1: LongMemEval-S (cleaned), real-style chat histories. Pre-registered in reports/Notebook reading at
shipping size 2026.md, amendment 1 (section 7), before any model call.

150 questions, 25 from each of the 6 question types (seed 0; abstention questions fall where they fall). Arms:
  full    all ~50 sessions of the history (~115K tokens), sorted by date
  oracle  only the evidence sessions (longmemeval_oracle.json)
  bm25    the 10 sessions a BM25 search over whole sessions returns for the question, sorted by date
Qwen3.5-27B runs oracle, bm25, full; Qwen3.5-4B runs oracle and bm25. Thinking on (16,384-token cap). Prompt, history
format (json) and judge prompts are LongMemEval's own (src/generation/run_generation.py, src/evaluation/evaluate_qa.py);
the has_answer markers are removed from every turn, as the official code does. Judge: gpt-4o-2024-08-06 when
OPENAI_API_KEY is set (the published judge); otherwise the endpoint at --judge-url, thinking off (a stated deviation).

  python brain/longmem.py --stub
  python brain/longmem.py --base-url URL27 --model Qwen/Qwen3.5-27B-FP8 --judge-url URL27
  python brain/longmem.py --base-url URL4 --model Qwen/Qwen3.5-4B --judge-url URL27 --judge-model Qwen/Qwen3.5-27B-FP8
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import threading
import time
import urllib.request
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import order_gap as G  # noqa: E402
import notebook_read as N  # noqa: E402  (ask, BM25, bootstrap, thinking settings)

ROOT = G.ROOT
BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"
FILES = {"longmemeval_s_cleaned.json": "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
         "longmemeval_oracle.json": "821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c"}
PER_TYPE, TOP_SESSIONS = 25, 10
LONG_HOLDS = 0.9           # full >= 0.9 x oracle: long context holds on real chat
TEMPLATE = ("I will give you several history chats between you and a user. Please answer the question based on the "
            "relevant chat history.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:")


def anscheck(task, question, answer, response, abstention=False):
    """LongMemEval's judge prompts, verbatim (src/evaluation/evaluate_qa.py)."""
    if abstention:
        return ("I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes "
                "if the model correctly identifies the question as unanswerable. The model could say that the information "
                "is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\n"
                "Explanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as "
                "unanswerable? Answer yes or no only.").format(question, answer, response)
    head = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the response "
            "contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or "
            "contains all the intermediate steps to get the correct answer, you should also answer yes. If the response "
            "only contains a subset of the information required by the answer, answer no. ")
    tail = "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
    if task in ("single-session-user", "single-session-assistant", "multi-session"):
        t = head + tail
    elif task == "temporal-reasoning":
        t = (head + "In addition, do not penalize off-by-one errors for the number of days. If the question asks for the "
             "number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the "
             "answer is 18), the model's response is still correct. " + tail)
    elif task == "knowledge-update":
        t = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the response "
             "contains the correct answer. Otherwise, answer no. If the response contains some previous information along "
             "with an updated answer, the response should be considered as correct as long as the updated answer is the "
             "required answer." + tail)
    elif task == "single-session-preference":
        t = ("I will give you a question, a rubric for desired personalized response, and a response from a model. Please "
             "answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to "
             "reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's "
             "personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model "
             "response correct? Answer yes or no only.")
    else:
        raise ValueError(task)
    return t.format(question, answer, response)


def load():
    out = {}
    for name, sha in FILES.items():
        path = os.path.join(ROOT, "data", "longmemeval", name)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(BASE + name, path)
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if h != sha:
            raise SystemExit(f"{path}: sha256 {h} is not the pre-registered {sha}")
        out[name] = {x["question_id"]: x for x in json.load(open(path, encoding="utf-8"))}
    return out["longmemeval_s_cleaned.json"], out["longmemeval_oracle.json"]


def sample(S):
    rng = random.Random(0)
    by = defaultdict(list)
    for qid in sorted(S):
        by[S[qid]["question_type"]].append(qid)
    return [qid for t in sorted(by) for qid in rng.sample(by[t], PER_TYPE)]


def clean(session):
    return [{k: v for k, v in turn.items() if k != "has_answer"} for turn in session]


def history(dates, sessions):
    chunks = sorted(zip(dates, sessions), key=lambda x: x[0])
    return "".join("\n### Session {}:\nSession Date: {}\nSession Content:\n{}\n".format(i + 1, d, "\n" + json.dumps(clean(s)))
                   for i, (d, s) in enumerate(chunks))


def prompt(entry, arm, bm_cache):
    if arm == "bm25":
        idx = bm_cache[entry["question_id"]]
        dates = [entry["haystack_dates"][i] for i in idx]
        sess = [entry["haystack_sessions"][i] for i in idx]
    else:
        dates, sess = entry["haystack_dates"], entry["haystack_sessions"]
    return TEMPLATE.format(history(dates, sess), entry["question_date"], entry["question"])


def bm25_sessions(entry):
    docs = [" ".join(t["content"] for t in s) for s in entry["haystack_sessions"]]
    bm = N.BM25(docs)
    q = [t for t in N.pnorm(entry["question"]).split() if t not in N.STOP]
    scores = []
    for i, (tf, d) in enumerate(zip(bm.tf, bm.toks)):
        x = sum(bm.idf.get(t, 0) * tf[t] * (bm.k1 + 1) / (tf[t] + bm.k1 * (1 - bm.b + bm.b * len(d) / bm.avg))
                for t in q if t in tf)
        scores.append((-x, i))
    return sorted(i for _, i in sorted(scores)[:TOP_SESSIONS])


class OpenAIJudge:
    model = "gpt-4o-2024-08-06"

    def __call__(self, text):
        body = json.dumps({"model": self.model, "messages": [{"role": "user", "content": text}], "temperature": 0,
                           "max_tokens": 10, "n": 1}).encode()
        for attempt in range(6):
            try:
                req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body, method="POST",
                                             headers={"Content-Type": "application/json",
                                                      "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]})
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.loads(r.read())["choices"][0]["message"]["content"]
            except Exception:  # noqa: BLE001  (rate limits and transient errors: back off, then give up loudly)
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)


class EndpointJudge:
    def __init__(self, url, model):
        self.llm = G.Endpoint(url, model, 1)
        self.model = self.llm.model + " (endpoint, thinking off)"

    def __call__(self, text):
        r = self.llm.chat(text, thinking=False, max_tokens=10, temperature=0)
        return r["choices"][0]["message"].get("content") or ""


class Stub:
    model = "stub"

    def __init__(self, S):
        self.gold = {e["question"]: e["answer"] for e in S.values()}

    def request(self, method, path, body):
        text = body["messages"][0]["content"]
        q = text.rsplit("Question: ", 1)[1].split("\nAnswer:")[0]
        n = text.count("### Session")
        ok = random.Random(zlib.crc32(text[-500:].encode()) + n).random() < 0.9 - 0.006 * n
        return {"choices": [{"message": {"content": "<think>stub</think>\n\n" + (str(self.gold.get(q, "")) if ok else
                                                                                   "I don't know.")},
                             "finish_reason": "stop"}], "usage": {"prompt_tokens": len(text) // 4, "completion_tokens": 5}}


def stub_judge(text):
    return "yes" if "Correct Answer: " in text and text.split("Correct Answer: ")[1].split("\n\nModel Response: ")[0] in \
        text.split("Model Response: ")[1] else "no"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-27B-FP8")
    p.add_argument("--base-url", default=None)
    p.add_argument("--judge-url", default=None)
    p.add_argument("--judge-model", default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--arms", default=None, help="default: oracle,bm25,full for 27B; oracle,bm25 for 4B")
    p.add_argument("--stub", action="store_true")
    args = p.parse_args()
    S, O = load()
    ids = sample(S)
    if args.stub:
        llm, tag, judge = Stub(S), "stub", stub_judge
    else:
        llm, tag = G.Endpoint(args.base_url, args.model, 1), args.model.split("/")[-1]
        judge = (OpenAIJudge() if os.environ.get("OPENAI_API_KEY") else
                 EndpointJudge(args.judge_url or args.base_url, args.judge_model or args.model))
    judge_name = getattr(judge, "model", "stub")
    arms = (args.arms.split(",") if args.arms else ["oracle", "bm25"] + ([] if "4B" in tag else ["full"]))
    out = os.path.join(ROOT, "results", f"longmem_{tag}")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "answers.jsonl")
    done = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            done[(r["arm"], r["question_id"])] = r
    bm_cache = {qid: bm25_sessions(S[qid]) for qid in ids}
    recall = np.mean([set(S[q]["answer_session_ids"]) <= {S[q]["haystack_session_ids"][i] for i in bm_cache[q]}
                      for q in ids])
    print(f"{len(ids)} questions; BM25 top-{TOP_SESSIONS} sessions hold all evidence sessions for {100 * recall:.0f}%; "
          f"judge {judge_name}; arms {arms}", flush=True)
    lock = threading.Lock()
    f = open(path, "a", encoding="utf-8")
    for arm in arms:
        src = O if arm == "oracle" else S
        todo = [q for q in ids if (arm, q) not in done]
        t0 = time.time()

        def one(qid, arm=arm, src=src):
            e = src[qid]
            text = prompt(e, arm, bm_cache)
            try:
                r = N.ask(llm, text, "think", zlib.crc32(f"{arm}|{qid}".encode()) % 2**31)
                err = None
            except SystemExit as x:                     # e.g. HTTP 400: longer than the endpoint's context
                r, err = {"answer": "", "truncated": False, "prompt_tokens": None, "completion_tokens": None,
                          "sec": None}, str(x)[:300]
            label = None
            if err is None:
                verdict = judge(anscheck(e["question_type"], e["question"], e["answer"], r["answer"],
                                         "_abs" in qid))
                label = "yes" in verdict.lower()
            rec = {"arm": arm, "question_id": qid, "type": e["question_type"], "abstention": "_abs" in qid,
                   "hypothesis": r["answer"], "label": label, "error": err, "chars": len(text),
                   **{k: r.get(k) for k in ("truncated", "prompt_tokens", "completion_tokens", "sec")}}
            with lock:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")
                f.flush()
                done[(arm, qid)] = rec

        with ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(one, todo))
        rs = [done[(arm, q)] for q in ids]
        ok = [r["label"] for r in rs if r["label"] is not None]
        print(f"{arm}: {100 * np.mean(ok):.1f}% of {len(ok)} judged, {sum(r['error'] is not None for r in rs)} errors, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    f.close()
    L = [f"# LongMemEval-S (cleaned), 150 questions: {tag}", "",
         f"Pre-registration: reports/Notebook reading at shipping size 2026.md, Part C. Judge: {judge_name}. Thinking on. "
         f"BM25 top-{TOP_SESSIONS} sessions hold all evidence sessions for {100 * recall:.0f}% of questions.", "",
         "| Type | " + " | ".join(arms) + " |", "|---" * (len(arms) + 1) + "|"]
    types = sorted({S[q]["question_type"] for q in ids})
    for t in types + ["all"]:
        cells = []
        for arm in arms:
            x = [done[(arm, q)]["label"] for q in ids if (arm, q) in done and done[(arm, q)]["label"] is not None
                 and (t == "all" or S[q]["question_type"] == t)]
            if x:
                m, lo, hi = N.boot_mean(x)
                cells.append(f"{m:.1f} [{lo:.0f}-{hi:.0f}] n={len(x)}")
            else:
                cells.append("-")
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L += ["", "## Verdict", ""]
    if all(a in arms for a in ("full", "oracle", "bm25")):
        both = [q for q in ids if all((a, q) in done and done[(a, q)]["label"] is not None for a in ("full", "oracle",
                                                                                                      "bm25"))]
        excl = len(ids) - len(both)
        F, Or, B = ([done[(a, q)]["label"] for q in both] for a in ("full", "oracle", "bm25"))
        rf = N.boot_ratio(F, Or)
        L.append(f"- full history {100 * np.mean(F):.1f}%, evidence only {100 * np.mean(Or):.1f}%, BM25 sessions "
                 f"{100 * np.mean(B):.1f}% on {len(both)} questions ({excl} excluded: errors such as context overflow). "
                 f"full/oracle {rf[0]:.2f} [{rf[1]:.2f}, {rf[2]:.2f}]: " +
                 ("LONG CONTEXT HOLDS on real chat (>= 0.9)." if rf[0] >= LONG_HOLDS else
                  "LONG CONTEXT DOES NOT HOLD on real chat (< 0.9).") +
                 (" Retrieval >= full history." if np.mean(B) >= np.mean(F) else " Retrieval < full history."))
    else:
        L.append("- (phone-sized model: oracle and BM25 only; no full-history verdict)")
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
