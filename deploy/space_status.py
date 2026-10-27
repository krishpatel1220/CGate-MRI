"""Check (and fix) the CGate-MRI Space: existence, build stage, visibility.

    .venv\\Scripts\\python.exe deploy\\space_status.py
    .venv\\Scripts\\python.exe deploy\\space_status.py --wait    (also wait up to 15 min for the build)

Prompts for your Hugging Face token locally (it is never stored), then:
  * reports whether the Space exists, its SDK, hardware and build stage;
  * prints the last lines of the cloud build log (so failures are visible);
  * if the Space is private, makes it public — a private Space cannot be
    opened by an evaluator, and has no visibility control in some UIs;
  * writes the same report to deploy\\space_status_last.txt, so it can be
    read back without copy-pasting anything.

Add --keep-private to report without changing visibility.
"""

from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
REPORT = WS / "deploy" / "space_status_last.txt"
LINES: list[str] = []


def say(text: str = "") -> None:
    print(text)
    LINES.append(text)


def main() -> int:
    """Always writes the report — including error paths — so the result can
    be read back without copy-pasting anything."""
    try:
        return _run()
    except SystemExit:
        raise
    except Exception as exc:  # crash: put the traceback in the report too
        import traceback

        say(f"CRASH   : {type(exc).__name__}: {exc}")
        say(traceback.format_exc())
        return 1
    finally:
        _flush()


def _run() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        say("huggingface_hub is missing. Run:")
        say("  .venv\\Scripts\\python.exe -m pip install -U huggingface_hub")
        return 1

    keep_private = "--keep-private" in sys.argv
    wait = "--wait" in sys.argv

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        token = getpass.getpass(
            "Hugging Face token (same one you used to deploy): "
        ).strip()
    if not token:
        say("token   : NONE GIVEN (press Enter pastes nothing — paste the token)")
        return 1

    api = HfApi(token=token)
    try:
        user = api.whoami().get("name")
    except Exception as exc:
        say(f"token   : REJECTED ({exc})")
        say("fix     : make a new token at https://huggingface.co/settings/tokens")
        say("          (role: write) and paste it at the prompt")
        return 1
    repo_id = f"{user}/cgate-mri"
    say(f"account : {user}")
    say(f"space   : {repo_id}")

    try:
        info = api.space_info(repo_id)
    except Exception as exc:
        say(f"space   : NOT FOUND ({exc})")
        say("fix     : run  .venv\\Scripts\\python.exe deploy\\create_space.py")
        return 1

    say(f"sdk     : {getattr(info, 'sdk', '?')}")
    say(f"private : {info.private}")

    def stage_and_endpoint() -> tuple[str, str]:
        try:
            rt = api.get_space_runtime(repo_id)
            stage = getattr(rt, "stage", "?")
            endpoint = getattr(rt, "endpoint", "") or ""
            return str(stage), str(endpoint)
        except Exception as exc:  # e.g. still queued
            return f"(runtime query failed: {exc})", ""

    stage, endpoint = stage_and_endpoint()
    say(f"stage   : {stage}")
    if endpoint:
        say(f"endpoint: {endpoint}")

    if wait and "running" not in stage.lower():
        say("waiting : up to 15 min for the build to finish ...")
        try:
            rt = api.wait_for_space(repo_id, timeout=900, poll_interval=10)
            stage = str(getattr(rt, "stage", stage))
            endpoint = str(getattr(rt, "endpoint", "") or endpoint)
            say(f"stage   : {stage}")
            if endpoint:
                say(f"endpoint: {endpoint}")
        except Exception as exc:
            say(f"waiting : gave up ({exc})")

    # Build logs: the reason a Space shows an error instead of the app.
    try:
        log_lines = list(api.fetch_space_logs(repo_id, build=True))
        if log_lines:
            say("")
            say("last build log lines:")
            for line in log_lines[-25:]:
                say("  | " + line.rstrip())
        else:
            say("logs    : (no build log yet)")
    except Exception as exc:
        say(f"logs    : could not fetch ({exc})")

    if info.private and not keep_private:
        say("")
        say("fixing  : making the Space public (private Spaces can't be shared) ...")
        try:
            api.update_repo_settings(repo_id, private=False, repo_type="space")
            say("fixing  : done -> now public")
        except Exception as exc:
            say(f"fixing  : FAILED ({exc})")
    elif info.private:
        say("fixing  : still private (--keep-private)")
    else:
        say("visibility: already public")

    say("")
    say(f"space page: https://huggingface.co/spaces/{repo_id}")
    say(f"demo url  : https://{repo_id.replace('/', '-').lower()}.hf.space")
    return 0


def _flush() -> None:
    try:
        REPORT.write_text("\n".join(LINES) + "\n", encoding="utf8")
        print(f"(report saved to {REPORT.relative_to(WS)})")
    except Exception as exc:
        print(f"(could not save report: {exc})")


if __name__ == "__main__":
    raise SystemExit(main())
