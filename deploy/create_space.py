"""Deploy CGate-MRI to Hugging Face Spaces — one command, free, always-on.

    .venv\\Scripts\\python.exe deploy\\create_space.py

What it does:
  1. reads a Hugging Face **write** token (prompted locally, or HF_TOKEN env
     var) — it is used for this run only and never written to disk;
  2. creates https://huggingface.co/spaces/<you>/cgate-mri (Docker SDK,
     public, free CPU hardware) if it does not exist yet;
  3. copies this project into the Space repository — the same file set as the
     GitHub repo, with the same exclusions (.venv, data/, caches, the 174 MB
     model …) — merges the Space's README front matter into README.md,
     commits and pushes;
  4. Hugging Face then builds the Dockerfile in the cloud (~10 minutes) and
     the demo is served at https://<you>-cgate-mri.hf.space

Rerun the script any time to ship an update; unchanged files are skipped.

First run only: create the token at https://huggingface.co/settings/tokens
(new token, role: **write**). No account yet? Sign up free first.
"""

from __future__ import annotations

import getpass
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
SPACE_NAME = "cgate-mri"

# Mirrors .gitignore: directories never copied into the Space repository.
EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "data",
    "Network-Intrusion-Detection-System",
    "__pycache__",
    ".pytest_cache",
    "_zip_staging",
    "cache",
    "table",
    "uploads",
    "test_uploads",
}
EXCLUDE_FILES = {"system_B6.pkl", "Thumbs.db", "Desktop.ini", ".DS_Store"}
EXCLUDE_SUFFIXES = (".zip", ".pyc")


def run(cmd: list[str], cwd: Path | None = None) -> str:
    """Run a command, die with its output if it fails."""
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"\nFAILED: {' '.join(cmd)}")
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit(1)
    return proc.stdout.strip()


def git(repo: Path, *args: str, token: str | None = None) -> str:
    """git in `repo`, authenticated via an extra header when a token is given."""
    cmd = ["git"]
    if token:
        cmd += ["-c", f"http.extraHeader=Authorization: Bearer {token}"]
    return run(cmd + list(args), cwd=repo)


def copy_tree(dest: Path) -> int:
    """Copy this project into the Space checkout. Returns files copied."""
    count = 0
    for root, dirs, files in os.walk(WS):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDE_DIRS)
        rel = Path(root).relative_to(WS)
        for name in sorted(files):
            if name in EXCLUDE_FILES or name.endswith(EXCLUDE_SUFFIXES):
                continue
            src = Path(root) / name
            dst = dest / rel / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            count += 1
    return count


def merge_readme(dest: Path) -> None:
    """Keep the Space's YAML front matter (sdk/app_port), replace the body."""
    existing = ""
    if (dest / "README.md").exists():
        existing = (dest / "README.md").read_text(encoding="utf8")

    kv: dict[str, str] = {}
    if existing.startswith("---"):
        end = existing.find("\n---", 3)
        if end != -1:
            for line in existing[3:end].splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    kv[key.strip()] = val.strip()

    # A Docker Space must declare these, and the body is ours.
    kv.setdefault("sdk", "docker")
    kv["app_port"] = "7860"
    kv.setdefault("title", "CGate-MRI")
    kv.setdefault("emoji", "\U0001f9ec")
    kv.setdefault("colorFrom", "blue")
    kv.setdefault("colorTo", "purple")
    kv.setdefault(
        "short_description",
        "Confidence-gated MRI analysis with grounded report generation",
    )

    front = "---\n" + "\n".join(f"{k}: {v}" for k, v in kv.items()) + "\n---\n"
    body = (WS / "README.md").read_text(encoding="utf8")
    (dest / "README.md").write_text(front + "\n" + body, encoding="utf8")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if shutil.which("git") is None:
        print("git was not found on PATH. Install it first (winget install Git.Git).")
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is missing. Run:")
        print("  .venv\\Scripts\\python.exe -m pip install -U huggingface_hub")
        return 1

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        token = getpass.getpass(
            "Hugging Face write token (https://huggingface.co/settings/tokens): "
        ).strip()
    if not token:
        print("No token given, nothing to do.")
        return 1

    api = HfApi(token=token)
    try:
        me = api.whoami()
    except Exception as exc:  # wrong/expired/insufficient token
        print(f"Token rejected: {exc}")
        print("Create a new token with role 'write' and try again.")
        return 1
    user = me.get("name") or me["fullname"]
    repo_id = f"{user}/{SPACE_NAME}"

    print(f"1. Ensuring Space {repo_id} exists (Docker SDK, public, cpu-basic) …")
    try:
        api.create_repo(
            repo_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            exist_ok=True,
            space_hardware="cpu-basic",
        )
    except TypeError:  # older huggingface_hub without space_hardware
        api.create_repo(
            repo_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True
        )
    except Exception as exc:
        print(f"   create_repo: {exc} (continuing — it may already exist)")

    print("2. Cloning the Space repository …")
    tmp = Path(tempfile.mkdtemp(prefix="hf-space-"))
    dest = tmp / SPACE_NAME
    clone_url = f"https://huggingface.co/spaces/{repo_id}"
    git(tmp, "clone", clone_url, str(dest), token=token)
    git(dest, "config", "user.name", user)
    git(dest, "config", "user.email", f"{user}@users.noreply.huggingface.co")

    print("3. Copying the project …")
    copied = copy_tree(dest)
    merge_readme(dest)
    print(f"   {copied} files staged")

    if git(dest, "status", "--porcelain"):
        print("4. Committing and pushing …")
        git(dest, "add", "-A")
        git(dest, "commit", "-m", "Deploy CGate-MRI web demo")
        git(dest, "push", token=token)
    else:
        print("4. Nothing changed — the Space repo is already up to date.")

    web = f"https://{repo_id.replace('/', '-').lower()}.hf.space"
    print("\nDone. The Space is now building in the cloud (~10 minutes):")
    print(f"  Space page (build log): {clone_url}")
    print(f"  Demo URL (after build): {web}")
    print("Watch the 'Building' log on the Space page; refresh the demo URL after.")
    print("The optional VLM is not installed there — the report page says so honestly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
