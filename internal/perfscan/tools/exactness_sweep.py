# exactness_sweep.py — find numeric store paths with NO tolerance-0 gate.
#
# For each source file it perturbs every store into a float slice by one ulp-scale
# factor, rebuilds, and runs the package suite. A file whose suite stays GREEN has no
# test sensitive to a 1-ulp change in that path.
#
# WHAT A GREEN RESULT MEANS, precisely: not "untested" — a gradcheck with a
# finite-difference tolerance legitimately cannot see one ulp. It means there is no
# EXACT gate, so any future rewrite of that path claiming bit-identity cannot be
# verified without first building an oracle. That build-the-gate-first cost is the
# thing this sweep exists to price, and it was learned by hitting it twice: the
# broadcast VJP and the prod VJP were both rewritten before anyone noticed no test
# could see the difference.
#
# A probe that DOES NOT COMPILE or matches nothing is reported as INVALID, never as
# green — a non-compiling mutation produces no failures and would otherwise read as
# proof of no coverage (PROC-MUTATION-VALID-001).
#
# Usage: python3 internal/perfscan/tools/exactness_sweep.py   (from the repo root)
import re, subprocess, shutil, sys, os
files = sorted(f for f in os.listdir("autograd") if f.startswith("vjp") and f.endswith(".go") and "_test" not in f)
results = []
for f in files:
    p = os.path.join("autograd", f)
    src = open(p).read()
    if "Storage().F64()" not in src and "Storage().F32()" not in src:
        continue
    # Perturb every store into a float slice by a 1-ulp-scale factor. Guaranteed to
    # compile (it is a float expression) and semantically visible to any exact gate.
    pat = re.compile(r"^(\t+)((?:ds|dst|acc|out|gi|gin)\w*\[[^\]]+\]) (=|\+=) (.+)$", re.M)
    new, cnt = pat.subn(lambda m: f"{m.group(1)}{m.group(2)} {m.group(3)} ({m.group(4)}) * 1.0000000000000002", src)
    if cnt == 0:
        results.append((f, 0, "no store matched"))
        continue
    shutil.copy(p, p + ".bak")
    open(p, "w").write(new)
    build = subprocess.run(["go", "build", "./autograd/"], capture_output=True)
    if build.returncode != 0:
        shutil.move(p + ".bak", p)
        results.append((f, cnt, "DID NOT COMPILE — invalid probe"))
        continue
    t = subprocess.run(["go", "test", "./autograd/", "-count", "1"], capture_output=True, text=True)
    fails = t.stdout.count("--- FAIL")
    shutil.move(p + ".bak", p)
    results.append((f, cnt, f"RED ({fails} failing)" if fails else "STILL GREEN — UNGATED"))
subprocess.run(["go", "build", "./autograd/"], capture_output=True)
for f, c, r in results:
    print(f"  {f:<28} stores={c:<3} {r}")
