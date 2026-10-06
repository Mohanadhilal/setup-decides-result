#!/usr/bin/env python3
"""
tools/check_repo.py
===================
Refuses to pass while the repository still contains anything that would embarrass
the paper: unfilled placeholders, internal notes, unpinned or missing dependencies,
an inconsistent or missing release tag, or missing result files.

Run from the repository root before every commit that the paper cites:

    python tools/check_repo.py --pre-commit  # before committing: everything but the tag
    python tools/check_repo.py            # after tagging
    python tools/check_repo.py --remote   # also checks the tag on GitHub

Exit code 0 means ready; anything else lists what must be fixed.
"""
import ast, os, re, subprocess, sys

TAG = "v1.0.2"
PLACEHOLDER = re.compile(r"<\s*FILL|\bFILL[\s_-]+IN\b|\bTODO\b|\bTBD\b|\bXXX\b", re.I)
TEXT_EXT = {".py", ".md", ".txt", ".yaml", ".yml", ".cff", ".json", ".csv", ".ps1", ".sh", ""}
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules"}
INTERNAL = ["docs/UPLOAD_CHECKLIST.md", "docs/PAPER_SECTION.md", "STEPS_AR.md"]
REQUIRED = ["README.md", "LICENSE", "CITATION.cff", "requirements.txt",
            "configs/baselines.yaml", "configs/training.yaml",
            "src/scaling.py", "src/sugar_extraction_env.py", "src/morl_score.py",
            "src/baseline_controllers.py",
            "scripts/compare_scaling.py", "scripts/reeval_protocol.py", "scripts/train_multiseed.py",
            "docs/ERRATA.md"]
REQUIRED_RESULTS = ["results/scaling/scaling_training.json",
                    "results/scaling/scaling_comparison.csv",
                    "results/scaling/scaling_verdict.txt"]
# import name -> pip distribution name
PIP = {"numpy": "numpy", "scipy": "scipy", "torch": "torch", "gymnasium": "gymnasium",
       "matplotlib": "matplotlib", "PIL": "pillow", "yaml": "pyyaml", "pandas": "pandas"}
STDLIB = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else set()

problems = []
def fail(msg): problems.append(msg)


def text_files():
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            p = os.path.join(root, f)[2:].replace("\\", "/")
            if os.path.splitext(f)[1].lower() in TEXT_EXT and os.path.getsize(p) < 5_000_000:
                yield p


# 1. placeholders anywhere (the checker itself is exempt: it names them on purpose)
for p in text_files():
    if p == "tools/check_repo.py": continue
    try: lines = open(p, encoding="utf-8-sig", errors="ignore").read().splitlines()
    except Exception: continue
    for n, line in enumerate(lines, 1):
        if PLACEHOLDER.search(line):
            fail(f"placeholder in {p}:{n}: {line.strip()[:80]}")

# 2. internal notes must not be public
for p in INTERNAL:
    if os.path.exists(p): fail(f"internal file still present: {p}")

# 3. required files
for p in REQUIRED + REQUIRED_RESULTS:
    if not os.path.exists(p): fail(f"missing: {p}")

# 3b. claims the paper and the letter make about the repository's contents
def nonempty_dir(p):
    return os.path.isdir(p) and any(os.scandir(p))
if not nonempty_dir("results/buggy_training_signal"):
    fail("missing or empty: results/buggy_training_signal (Section 5.8 and the letter say these runs are kept)")
enum = [f for d in ("scripts", "results") if os.path.isdir(d) for r, _, fs in os.walk(d) for f in fs
        if re.search(r"pareto|attainable|enumerat", f, re.I)]
if not enum:
    fail("no Pareto-front enumeration script or output found (the letter lists it under R1-9)")

# 4. requirements: every third-party import covered, every line pinned
req = {}
if os.path.exists("requirements.txt"):
    for line in open("requirements.txt", encoding="utf-8-sig"):
        line = line.strip()
        if not line or line.startswith("#"): continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*==\s*([0-9][^\s;]*)", line)
        if not m: fail(f"requirements.txt: not pinned with '==': {line}")
        else: req[m.group(1).lower()] = m.group(2)
used = set()
for d in ("src", "scripts"):
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith(".py"):
                try: tree = ast.parse(open(os.path.join(root, f), encoding="utf-8-sig").read())
                except SyntaxError as e: fail(f"syntax error in {root}/{f}: {e}"); continue
                for node in ast.walk(tree):
                    names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                             else [node.module] if isinstance(node, ast.ImportFrom) and node.module and node.level == 0
                             else [])
                    for nm in names: used.add(nm.split(".")[0])
for mod in sorted(used):
    if mod in PIP and PIP[mod] not in req:
        fail(f"requirements.txt lacks '{PIP[mod]}' (imported as '{mod}')")

# 5. tag consistency in text: the paper and README cite v1.0.2, nothing else
for p in ("README.md", "CITATION.cff"):
    if os.path.exists(p):
        s = open(p, encoding="utf-8-sig").read()
        if "paper-v1" in s: fail(f"{p} still names the old tag 'paper-v1'")
        if TAG not in s: fail(f"{p} does not name the tag {TAG}")

# 6. baselines.yaml is complete
try:
    import yaml
    b = yaml.safe_load(open("configs/baselines.yaml", encoding="utf-8"))
    if len(b["fuzzy"]["rule_base"]) != 9: fail("baselines.yaml: fuzzy rule base is not nine rules")
    if b["mpc"]["internal_model"]["time_constant_s"] != 540: fail("baselines.yaml: MPC time constant is not 540 s")
except ImportError:
    print("note: install pyyaml to validate configs/baselines.yaml")
except Exception as e:
    fail(f"configs/baselines.yaml unreadable or incomplete: {e}")

# 7. the tag exists locally, and on GitHub if asked
def git(*a):
    r = subprocess.run(["git", *a], capture_output=True, text=True)
    return r.returncode, r.stdout.strip()
rc, tags = git("tag", "-l", TAG)
if rc != 0: fail("not a git repository, or git unavailable")
elif TAG not in tags.split() and "--pre-commit" not in sys.argv:
    fail(f"tag {TAG} does not exist locally")
if "--remote" in sys.argv:
    rc, out = git("ls-remote", "--tags", "origin")
    if rc != 0: fail("could not reach origin")
    elif f"refs/tags/{TAG}" not in out: fail(f"tag {TAG} is not on GitHub: run  git push origin {TAG}")
    else:
        rc, local = git("rev-list", "-n", "1", TAG)
        print(f"tag {TAG} -> commit {local}")
        # The repository's front page shows the default branch, not the tag. If GitHub's main is
        # behind the tag, a reviewer opening the URL in the paper still sees the old files.
        rc2, heads = git("ls-remote", "origin", "refs/heads/main")
        remote_main = heads.split()[0] if heads else ""
        if remote_main != local:
            fail(f"GitHub's main is at {remote_main[:7] or 'nothing'}, not at the tagged commit {local[:7]}: "
                 f"the repository front page still shows the old files. Push the branch (Push origin).")
        else:
            print(f"GitHub main -> {remote_main}  (matches the tag)")

print("=" * 72)
if problems:
    print(f"NOT READY: {len(problems)} problem(s)")
    for p in problems: print("  -", p)
    sys.exit(1)
if "--pre-commit" in sys.argv:
    print("READY TO COMMIT: no placeholders, no internal files, dependencies pinned and complete (tag not yet checked)")
else:
    print(f"READY: no placeholders, no internal files, dependencies pinned and complete, tag {TAG} present")
rc, h = git("rev-parse", "HEAD"); print(f"HEAD commit: {h}")
