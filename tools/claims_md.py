"""Regenerate CLAIMS.md from claims.py and the recorded reference outputs.

    python tools/claims_md.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from claims import CLAIMS

RESULTS = {"benchmark_todd": "results/benchmark/todd_seed4_suite.log",
           "benchmark_teleport": "results/benchmark/teleport_suite.log"}


def status(c):
    if os.path.exists(os.path.join(ROOT, "expected", c["id"] + ".txt")):
        return "reference recorded"
    if c["id"] in RESULTS:
        return "paper's run in `" + RESULTS[c["id"]] + "`"
    return "none (full tier)" if c["tier"] == "full" else "missing"


lines = ["# Claims index", "",
         "Every computational result in the paper, the command that reproduces it, and its tier. Run one with",
         "`python reproduce.py <id>`, or a whole tier with `python reproduce.py --tier quick|standard|full`.",
         "This file is generated from `claims.py` by `tools/claims_md.py`.", "",
         "| id | paper labels | App. | what it checks | command | tier | reference |",
         "|---|---|---|---|---|---|---|"]
for c in CLAIMS:
    cmd = " ".join(c["cmd"]).replace("|", "\\|")
    lines.append(f"| `{c['id']}` | {c['labels']} | {c['app']} | {c['what']} | `{cmd}` | {c['tier']} | {status(c)} |")
n = {t: sum(1 for c in CLAIMS if c["tier"] == t) for t in ("quick", "standard", "full")}
rec = sum(1 for c in CLAIMS if status(c) == "reference recorded")
lines += ["", f"{len(CLAIMS)} claims: {n['quick']} quick, {n['standard']} standard, {n['full']} full; "
          f"{rec} have a recorded reference output."]
open(os.path.join(ROOT, "CLAIMS.md"), "w", encoding="utf-8", newline="").write("\n".join(lines) + "\n")
print("\n".join(lines[-1:]))
