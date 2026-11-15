"""Round 2: how does the 4B fail one-pass (thinking-off) two-hop? Reads test 1's saved answers (no model call)."""
import json, collections
rows = [json.loads(l) for l in open("/home/user/nichtec/results/notebook_read_Qwen3.5-4B/answers.jsonl")]
for lvl in (10, 100, 5000):
    for mode in ("nothink", "nothink256", "think"):
        w = [a for a in rows if a["kind"] == "two" and a["mode"] == mode and a["level"] == lvl]
        if w:
            bad = [a for a in w if not a["correct"]]
            print(lvl, mode, "n", len(w), "right", len(w) - len(bad), collections.Counter(a.get("source") for a in bad).most_common(4))
