"""Setup orchestrator — prepares the whole project so app.py is ready to run.

Runs every step in order:
  1. create required folders (models/, results/)
  2. train + save the Logistic Regression model
  3. download + save the BERT model
  4. compare both models -> results CSVs
  5. build the aspect-sentiment table

After this finishes successfully, just run:
    python3 app.py

Usage:
    python3 scripts.py                # run everything
    python3 scripts.py --skip-bert    # skip the heavy BERT download
    python3 scripts.py --force        # re-run even if models already exist
"""
import os
import sys
import time
import subprocess

# ---- paths ----
ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "artifact")
LOGREG_FILE = os.path.join(MODELS_DIR, "logreg_model.joblib")
BERT_DIR = os.path.join(MODELS_DIR, "bert_sentiment")

# ---- CLI flags ----
SKIP_BERT = "--skip-bert" in sys.argv
FORCE = "--force" in sys.argv


def log(msg):
    print(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}")


def run_step(name, command):
    """Run a shell command, stream output, stop the whole setup on failure."""
    log(name)
    start = time.time()
    result = subprocess.run(command, shell=True, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n[FAILED] {name} (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"[OK] {name}  ({time.time() - start:.1f}s)")


def ensure_folders():
    log("Step 1/5 — Creating folders")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"[OK] {MODELS_DIR}")
    print(f"[OK] {RESULTS_DIR}")


def maybe_install():
    """Install requirements if a requirements.txt is present."""
    req = os.path.join(ROOT, "requirements.txt")
    if os.path.exists(req):
        run_step("Step 0 — Installing requirements",
                 f'"{sys.executable}" -m pip install -r requirements.txt')


def main():
    log("SENTIMENT PROJECT SETUP")
    py = f'"{sys.executable}"'

    # optional dependency install
    maybe_install()

    # 1) folders
    ensure_folders()

    # 2) baseline model
    if FORCE or not os.path.exists(LOGREG_FILE):
        run_step("Step 2/5 — Training Logistic Regression model",
                 f"{py} src/train_logreg.py")
    else:
        print("\n[skip] logreg model already exists (use --force to retrain)")

    # 3) BERT model
    if SKIP_BERT:
        print("\n[skip] BERT training skipped (--skip-bert)")
    elif FORCE or not os.path.isdir(BERT_DIR):
        run_step("Step 3/5 — Downloading + saving BERT model",
                 f"{py} src/train_bert.py")
    else:
        print("\n[skip] BERT model already exists (use --force to re-download)")

    # 4) comparison
    run_step("Step 4/5 — Comparing models -> results CSVs",
             f"{py} src/compare_models.py")

    # 5) aspects
    run_step("Step 5/5 — Building aspect-sentiment table",
             f"{py} src/aspects.py")

    log("SETUP COMPLETE ✅")
    print("Environment is ready. Now run:\n")
    print("    python3 app.py\n")
    print("Or launch the dashboard:\n")
    print("    streamlit run src/dashboard.py\n")


if __name__ == "__main__":
    main()
