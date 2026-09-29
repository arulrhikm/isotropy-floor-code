"""Reproduce the paper's computational claims.

    python reproduce.py --list                  every claim, its paper labels and tier
    python reproduce.py atlas incremental       run named claims
    python reproduce.py --tier quick            claims that each run in about a minute or less
    python reproduce.py --tier standard         quick plus claims that take up to about 30 minutes each
    python reproduce.py --tier full             everything, including the benchmark suites (hours)

Each claim runs in its own process.  Its output goes to out/<id>.txt and is compared with
expected/<id>.txt after removing wall-clock times; the run fails if a claim's command fails or its
output differs.  Full-tier claims have no expected/ file: their recorded output is in results/,
and they are reported as NO-REFERENCE rather than failed.  `--update-expected` rewrites expected/
from the current run.
"""
import argparse, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from claims import CLAIMS

ORDER = {"quick": 0, "standard": 1, "full": 2}
TIMING = [
    re.compile(r"\b\d+\.\d+\s*(ms|s|sec|secs|seconds|min)\b"),         # "0.87 s", "1.2s"
    re.compile(r"\b\d+\s*(ms|sec|secs|seconds|min)\b"),                 # "12 sec", "3 min"
    re.compile(r"\b\d+ s\b"),                                          # "12 s" (not "2s")
    re.compile(r"\(\d+s\)"),                                          # "(454s)"
    re.compile(r"\b(time|elapsed|took|wall)\b[^\n]*", re.I),           # whole clauses reporting time
]


def normalize(text, claim):
    lines = []
    for ln in text.replace("\r\n", "\n").split("\n"):
        if claim["id"].startswith("benchmark_"):
            p = ln.split()
            if len(p) >= 15 and p[1].isdigit():
                p[11] = p[12] = "<t>"                               # floor_s, pyzx_s columns
                ln = " ".join(p)
        for rx in TIMING:
            ln = rx.sub("<t>", ln)
        lines.append(ln.rstrip())
    return "\n".join(lines).strip() + "\n"


def run(claim, update):
    t0 = time.time()
    out_dir, exp_dir = os.path.join(ROOT, "out"), os.path.join(ROOT, "expected")
    os.makedirs(out_dir, exist_ok=True)
    env = {**os.environ, "PYTHONHASHSEED": "0", "ISOFLOOR_BENCH": os.path.join(ROOT, "benchmarks", "qc"),
           "ISOFLOOR_OUT": out_dir, "PYTHONIOENCODING": "utf-8"}
    cmd = [sys.executable if c == "python" else c for c in claim["cmd"]]
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=claim["timeout"], env=env,
                           encoding="utf-8", errors="replace")
        text, rc = p.stdout + (("\n[stderr]\n" + p.stderr) if p.returncode else ""), p.returncode
    except subprocess.TimeoutExpired:
        text, rc = "[timeout]\n", -1
    dt = time.time() - t0
    open(os.path.join(out_dir, claim["id"] + ".txt"), "w", encoding="utf-8", newline="\n").write(text)
    exp = os.path.join(exp_dir, claim["id"] + ".txt")
    if update and rc == 0:
        os.makedirs(exp_dir, exist_ok=True)
        open(exp, "w", encoding="utf-8", newline="\n").write(normalize(text, claim))
        status = "RECORDED"
    elif rc != 0:
        status = "FAILED" if rc != -1 else "TIMEOUT"
    elif not os.path.exists(exp):
        status = "NO-REFERENCE" if claim["tier"] == "full" else "MISSING"
    else:
        ref = open(exp, encoding="utf-8").read().replace("\r\n", "\n")
        status = "OK" if normalize(text, claim) == ref else "DIFFERS"
    print(f"{status:12s} {claim['id']:24s} {dt:8.1f} s   {claim['labels']}", flush=True)
    return status in ("OK", "RECORDED", "NO-REFERENCE")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--tier", choices=list(ORDER))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--update-expected", action="store_true")
    a = ap.parse_args()
    if a.list or not (a.ids or a.tier):
        for c in CLAIMS:
            print(f"{c['id']:24s} {c['tier']:9s} App. {c['app']:8s} {c['labels']}\n{'':24s} {c['what']}")
        return 0
    chosen = [c for c in CLAIMS if c["id"] in a.ids] if a.ids else [c for c in CLAIMS if ORDER[c["tier"]] <= ORDER[a.tier]]
    unknown = set(a.ids) - {c["id"] for c in CLAIMS}
    if unknown:
        print("unknown claim ids:", ", ".join(sorted(unknown))); return 2
    ok = [run(c, a.update_expected) for c in chosen]
    print(f"{sum(ok)}/{len(ok)} claims reproduced")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
