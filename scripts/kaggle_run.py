"""Push a notebook to Kaggle, wait for it, read its log (P0 infra).

Kaggle is the project's primary compute, but a session is ephemeral and headless: there
is no endpoint to call into and nothing survives except `/kaggle/working`. So the loop is
push a notebook, poll until it stops, read the log. This wraps that loop.

**Why the log is the real output channel.** `/kernels/output` lists files one page at a
time, and the first page is swamped by the thousands of hashed cache entries a run
leaves behind -- a `results/<date>_<name>/metrics.json` sits far past the end of it. So
every runner prints its metrics to stdout as well as writing them to disk, and `log`
below is how results actually come back. `fetch` is for when a specific file is needed
by name and the path is already known.

    python -m scripts.kaggle_run push  notebooks/kaggle_c1.ipynb --slug verimem-c1
    python -m scripts.kaggle_run wait  verimem-c1
    python -m scripts.kaggle_run log   verimem-c1 --grep ASR
    python -m scripts.kaggle_run fetch verimem-c1 results/2026-10-28_x/metrics.json

Auth is the token Kaggle's own "copy" button writes to `~/.kaggle/access_token`. It is
read here and sent as a bearer header; it is never printed, logged or committed (rule 9).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

API = "https://www.kaggle.com/api/v1"
TOKEN_PATHS = ("~/.kaggle/access_token", "~/.kaggle/kaggle.json")
# A kernel that has stopped is in one of these; anything else means still working.
TERMINAL = {"complete", "error", "cancelAcknowledged", "cancelRequested"}


def _token() -> str:
    """The bearer token, from wherever Kaggle's copy button put it."""
    for raw in TOKEN_PATHS:
        p = Path(os.path.expanduser(raw))
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8").strip()
        if raw.endswith(".json"):
            # kaggle.json is {username, key}; the key is not a bearer token, so say so
            # rather than sending it and getting an opaque 401.
            raise SystemExit(
                f"{p} holds a username/key pair, not an access token. Open a Kaggle "
                "notebook, use its token copy-command, and re-run."
            )
        if text:
            return text
    raise SystemExit(
        "No Kaggle token found. Expected one of: "
        + ", ".join(TOKEN_PATHS)
        + "\nKaggle's notebook UI has a copy-command that writes it for you."
    )


def _call(path: str, payload: dict[str, Any] | None = None, timeout: int = 300) -> dict[str, Any]:
    url = f"{API}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {_token()}",
            "Content-Type": "application/json",
        },
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:500]
        raise SystemExit(f"Kaggle API {exc.code} on {path}\n{body}") from exc


def push(notebook: Path, slug: str, owner: str, datasets: list[str], enable_gpu: bool) -> str:
    """Upload a notebook and start it. Returns the kernel's full slug."""
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    meta = {
        "id": f"{owner}/{slug}",
        "title": slug,
        "code_file": notebook.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_internet": True,
        "dataset_sources": datasets,
        "competition_sources": [],
        "kernel_sources": [],
    }
    body = {**meta, "text": json.dumps(nb)}
    out = _call("/kernels/push", body)
    if out.get("error"):
        raise SystemExit(f"push rejected: {out['error']}")
    print(f"[kaggle] pushed {owner}/{slug} -> {out.get('url')}")
    return f"{owner}/{slug}"


def status(slug: str, owner: str) -> dict[str, Any]:
    q = urllib.parse.urlencode({"userName": owner, "kernelSlug": slug})
    return _call(f"/kernels/status?{q}", timeout=60)


def wait(slug: str, owner: str, poll: int = 30, limit: int = 43200) -> str:
    """Poll until the kernel reaches a terminal state.

    Matches terminal states positively rather than testing "not running": an empty or
    unexpected status string would otherwise read as finished and the caller would go on
    to read a log that is not there yet.
    """
    start = time.time()
    last = ""
    while time.time() - start < limit:
        st = str(status(slug, owner).get("status") or "")
        if st != last:
            print(f"[kaggle] {int(time.time() - start):>6}s  {st}", flush=True)
            last = st
        if st in TERMINAL:
            return st
        time.sleep(poll)
    raise SystemExit(f"{slug} still {last!r} after {limit}s; check the notebook in the UI")


def log(slug: str, owner: str, grep: str | None, tail: int) -> str:
    """The kernel's stdout. This is where results come back from; see the module docstring."""
    q = urllib.parse.urlencode({"userName": owner, "kernelSlug": slug})
    out = _call(f"/kernels/output?{q}")
    lines = [(e.get("data") or "").rstrip() for e in json.loads(out.get("log") or "[]")]
    if grep:
        lines = [ln for ln in lines if grep in ln]
    # Kaggle duplicates stream records for a cell that writes to both stdout and the
    # notebook display, so identical consecutive lines are an artefact, not data.
    deduped = [ln for i, ln in enumerate(lines) if i == 0 or ln != lines[i - 1]]
    return "\n".join(deduped[-tail:] if tail else deduped)


def fetch(slug: str, owner: str, name: str, dest: Path) -> Path:
    """Download one output file by its path inside /kaggle/working."""
    q = urllib.parse.urlencode({"userName": owner, "kernelSlug": slug, "fileName": name})
    url = f"{API}/kernels/output/download?{q}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {_token()}"})
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(req, timeout=600) as r, dest.open("wb") as fh:
            fh.write(r.read())
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"Kaggle API {exc.code} fetching {name!r}. The path must be exactly as it "
            "appears under /kaggle/working. Results also print to the log -- try `log`."
        ) from exc
    print(f"[kaggle] {name} -> {dest} ({dest.stat().st_size} bytes)")
    return dest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m scripts.kaggle_run")
    p.add_argument("--owner", default=os.environ.get("KAGGLE_USERNAME", "teesha5124"))
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("push", help="upload a notebook and start it")
    sp.add_argument("notebook", type=Path)
    sp.add_argument("--slug", required=True)
    sp.add_argument("--dataset", action="append", default=[], help="repeatable owner/name")
    sp.add_argument("--no-gpu", dest="gpu", action="store_false")
    sp.add_argument(
        "--wait", action="store_true", help="block until it finishes, then print the log"
    )

    sp = sub.add_parser("status", help="one status check")
    sp.add_argument("slug")

    sp = sub.add_parser("wait", help="poll until the kernel stops")
    sp.add_argument("slug")
    sp.add_argument("--poll", type=int, default=30)

    sp = sub.add_parser("log", help="print the kernel's stdout")
    sp.add_argument("slug")
    sp.add_argument("--grep", default=None)
    sp.add_argument("--tail", type=int, default=0, help="0 means all")

    sp = sub.add_parser("fetch", help="download one output file by path")
    sp.add_argument("slug")
    sp.add_argument("name")
    sp.add_argument("--out", type=Path, default=None)

    args = p.parse_args(argv)

    if args.cmd == "push":
        push(args.notebook, args.slug, args.owner, args.dataset, args.gpu)
        if args.wait:
            st = wait(args.slug, args.owner)
            print(log(args.slug, args.owner, None, 0))
            return 0 if st == "complete" else 1
        return 0
    if args.cmd == "status":
        print(json.dumps(status(args.slug, args.owner), indent=2))
        return 0
    if args.cmd == "wait":
        st = wait(args.slug, args.owner, poll=args.poll)
        print(f"[kaggle] {args.slug}: {st}")
        return 0 if st == "complete" else 1
    if args.cmd == "log":
        print(log(args.slug, args.owner, args.grep, args.tail))
        return 0
    if args.cmd == "fetch":
        fetch(args.slug, args.owner, args.name, args.out or Path(Path(args.name).name))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
