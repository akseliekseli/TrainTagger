#!/usr/bin/env python3
"""Check EOS chunks directly; optionally recover readable copies from the mount.

Run on lxplus before training, with dataset writers and other repair runs stopped.
Reads both splits' metadata without modifying it. Compression-header errors,
zlib decompression errors, and jet-count mismatches are eligible for recovery.
No chunks are deleted or relabelled. Originals are renamed into .chunk-recovery.
Requires uproot, xrdcp, xrdfs and valid EOS credentials.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from uuid import uuid4
import zlib

import uproot


class JetCountMismatch(ValueError):
    """A readable chunk does not contain the number of jets in metadata."""


def command(*args, timeout=300):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{' '.join(map(str, args))}\n{result.stderr or result.stdout}")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(path, expected):
    # Read every field, including every compressed page, just as training does.
    # Only one chunk is held in memory at a time.
    with uproot.open(path, object_cache=None, array_cache=None) as root_file:
        data = root_file["data"].arrays(library="ak")
        if len(data) != expected:
            raise JetCountMismatch(f"Jet count is {len(data)}; metadata says {expected}")


def manifest_chunks(dataset):
    chunks = []
    seen = set()
    for split in ("training_data", "testing_data"):
        manifest = dataset / split / "metadata.json"
        with manifest.open() as stream:
            entries = json.load(stream)
        if not isinstance(entries, list):
            raise ValueError(f"Expected a list in {manifest}")
        for entry in entries:
            path = Path(entry["file"])
            if not path.is_absolute():
                raise ValueError(f"Expected an absolute metadata path: {path}")
            path = Path(os.path.abspath(path))  # Do not resolve the EOS mount.
            if path.parent != dataset / split:
                raise ValueError(f"Chunk outside its split directory: {path}")
            if path in seen:
                raise ValueError(f"Duplicate metadata entry: {path}")
            count = entry["entries"]
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError(f"Invalid metadata jet count for {path}: {count}")
            seen.add(path)
            chunks.append((path, count, split))
    if not chunks:
        raise ValueError("No chunks in the two metadata files")
    return chunks


def recover(args, path, expected, remote, failed_copy, scratch, backup_dir):
    def url(remote_path):
        return f"root://{args.eos_host}/{remote_path}"

    def download(remote_path, destination):
        command("xrdcp", "--nopbar", url(remote_path), str(destination), timeout=args.timeout)

    def move(source, destination):
        command("xrdfs", args.eos_host, "mv", str(source), str(destination), timeout=args.timeout)

    # Snapshot the readable mounted copy. Retain it locally if recovery fails.
    candidate = scratch / "candidate.root"
    shutil.copyfile(path, candidate)
    validate(candidate, expected)
    candidate_hash = sha256(candidate)
    failed_hash = sha256(failed_copy)
    if candidate_hash == failed_hash:
        raise RuntimeError("Mounted and failing copies have identical bytes; cannot repair this way")


    staged = backup_dir / f"{path.name}.replacement"
    backup = backup_dir / f"{path.name}.original"
    command("xrdfs", args.eos_host, "mkdir", "-p", str(backup_dir), timeout=args.timeout)
    command("xrdcp", "--nopbar", str(candidate), url(staged), timeout=args.timeout)
    print(f"  Staged recovery: {url(staged)}", flush=True)

    # Verify the upload through XRootD before touching the original filename.
    uploaded = scratch / "uploaded.root"
    download(staged, uploaded)
    if sha256(uploaded) != candidate_hash:
        raise RuntimeError("Uploaded recovery does not match the readable local copy")
    validate(uploaded, expected)

    # Detect changes since the failing read; do not replace a moving target.
    recheck = scratch / "original_recheck.root"
    download(remote, recheck)
    if sha256(recheck) != failed_hash:
        raise RuntimeError("EOS contents changed between reads; original left untouched")

    print(f"  Preserving original at: {url(backup)}", flush=True)
    move(remote, backup)
    installed = False
    try:
        move(staged, remote)
        installed = True
        final_copy = scratch / "final.root"
        download(remote, final_copy)
        if sha256(final_copy) != candidate_hash:
            raise RuntimeError("Final original-name download has a different SHA-256")
        validate(final_copy, expected)
    except BaseException as error:
        # Also handle Ctrl-C during installation. The backup always remains
        # available if an interrupted/ambiguous remote operation prevents rollback.
        try:
            if installed:
                move(remote, backup_dir / f"{path.name}.failed-replacement")
            move(backup, remote)
        except BaseException as rollback_error:
            raise RuntimeError(
                f"Recovery failed ({error}); rollback also failed ({rollback_error}). "
                f"Inspect original {url(remote)}, backup {url(backup)}, "
                f"staged copy {url(staged)}, and local candidate {candidate} before training."
            ) from error
        raise
    return {"backup": url(backup), "sha256": candidate_hash}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--repair", action="store_true", help="Recover matching failures; default is scan only")
    parser.add_argument("--eos-host", default="eoshome-a.cern.ch")
    parser.add_argument("--mount-prefix", type=Path, default=Path("/eos/home-a"))
    parser.add_argument("--eos-prefix", type=PurePosixPath, default=PurePosixPath("/eos/user/a"))
    parser.add_argument("--scratch-dir", type=Path, default=Path("/tmp"), help="Local disk, not an EOS mount")
    parser.add_argument("--timeout", type=int, default=300, help="Seconds per XRootD command")
    args = parser.parse_args(argv)
    for executable in ("xrdcp", "xrdfs"):
        if shutil.which(executable) is None:
            parser.error(f"{executable} is not on PATH")
    dataset = Path(os.path.abspath(args.data_dir))
    mount = Path(os.path.abspath(args.mount_prefix))
    remote_dataset = args.eos_prefix / dataset.relative_to(mount).as_posix()
    chunks = manifest_chunks(dataset)  # Validate the manifest before any writes.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:12]
    workspace = Path(tempfile.mkdtemp(prefix="sc8-repair-", dir=args.scratch_dir))
    print(f"Uproot {uproot.__version__}; {len(chunks)} chunks; repair={args.repair}", flush=True)
    print(f"Local report and any unrecovered candidates: {workspace}", flush=True)
    results = []

    for number, (path, expected, split) in enumerate(chunks, 1):
        remote = args.eos_prefix / path.relative_to(mount).as_posix()
        scratch = workspace / split / path.stem
        scratch.mkdir(parents=True)
        downloaded = scratch / "from_xrootd.root"
        item = {"file": str(path), "status": "unresolved"}
        print(f"[{number}/{len(chunks)}] {split}/{path.name}", flush=True)
        try:
            # Transfer/authentication errors must not trigger a replacement.
            command("xrdcp", "--nopbar", f"root://{args.eos_host}/{remote}",
                    str(downloaded), timeout=args.timeout)
            try:
                validate(downloaded, expected)
            except Exception as error:
                item["read_error"] = str(error)
                recoverable = isinstance(error, (zlib.error, JetCountMismatch)) or (
                    isinstance(error, ValueError)
                    and "unrecognized compression algorithm" in str(error)
                )
                if not recoverable:
                    raise
                if not args.repair:
                    raise RuntimeError(f"Chunk validation failed; scan only: {error}") from error
                print(f"  {error}\n  Checking the mounted copy for recovery", flush=True)
                backup_dir = remote_dataset / ".chunk-recovery" / run_id / split
                item.update(recover(args, path, expected, remote, downloaded, scratch, backup_dir))
                item["status"] = "repaired"
                print("  REPAIRED and verified under its original filename", flush=True)
            else:
                item["status"] = "ok"
            # Only remove our own temporary downloads after a successful check.
            shutil.rmtree(scratch)
        except Exception as error:
            item["error"] = str(error)
            print(f"  UNRESOLVED: {error}\n  Local evidence retained: {scratch}", flush=True)
        results.append(item)
        # Keep a report even if the scan is interrupted later.
        with (workspace / "report.json").open("w") as stream:
            json.dump(results, stream, indent=2)

    counts = {status: sum(item["status"] == status for item in results)
              for status in ("ok", "repaired", "unresolved")}
    print(f"Summary: {counts}\nReport: {workspace / 'report.json'}", flush=True)
    print("Metadata and jet labels were not changed.", flush=True)
    return 1 if counts["unresolved"] else 0


if __name__ == "__main__":
    sys.exit(main())
