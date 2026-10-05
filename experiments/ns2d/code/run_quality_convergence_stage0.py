"""QUALITY_CONVERGENCE_CONFIRM_V1 Stage 0 -- fetch the held-out shards and seal the cohort.

Downloads the sixteen `valid` and `test` shards at a pinned revision, hashes all twenty-four
cohort files as raw bytes, checks every digest against the publisher's own LFS checksums, and
writes a create-only data seal.

`h5py` is never imported by this module. That is the mechanism, not a coincidence: hashing a
file's bytes is not opening its contents, and the two must not be conflated. The seal therefore
records two independent counters,

    raw_bytes_hashed    train 8, selection 8, test 8
    hdf5_field_reads    train 0, selection 0, test 0

and it is `hdf5_field_reads` that Stage A through D must keep at zero for TEST. A protocol whose
`test_files_opened` counter is tripped by a checksum would be contradicting itself.

Nothing here reads a shape, a dataset name or a field. Schema checks belong to the stage at which
each role is first legitimately opened.

Protocol: `protocols/QUALITY_CONVERGENCE_CONFIRM_V1.md`, bound by digest in the seal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import sys
from datetime import datetime, timezone

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

PROTOCOL_ID = "QUALITY_CONVERGENCE_CONFIRM_V1"
PROTOCOL_PATH = "protocols/QUALITY_CONVERGENCE_CONFIRM_V1.md"
STAGE = 0

HF_REPO_ID = "pdearena/NavierStokes-2D"
HF_REVISION = "cd99556a883a20acb9102c1f4bfdfae66ae33495"

#: Fixed by the protocol: use the first eight eligible entries in ascending order.
FAMILIES = (61749, 61991, 97473, 433143, 445636, 452245, 453445, 471651)

VALID_OFFSET = 100
TEST_OFFSET = 200
TRAIN_BYTES = 550_726_984
HELDOUT_BYTES = 137_685_784


def expected_files():
    """The twenty-four cohort filenames, by role, derived from the frozen family list."""
    rows = []
    for family in FAMILIES:
        rows.append(("train", family,
                     f"NavierStokes2D_train_{family}_0.50000_100.h5", TRAIN_BYTES))
        rows.append(("selection", family,
                     f"NavierStokes2D_valid_{family + VALID_OFFSET}_0.50000.h5", HELDOUT_BYTES))
        rows.append(("test", family,
                     f"NavierStokes2D_test_{family + TEST_OFFSET}_0.50000.h5", HELDOUT_BYTES))
    return rows


def download_patterns():
    """Only the sixteen held-out shards. The eight train shards are already local."""
    return [name for role, _, name, _ in expected_files() if role != "train"]


def sha256_file(path, block=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_publisher_digests(path):
    """{filename: {sha256, bytes}} read from the pinned revision's LFS pointers.

    Verifying against the publisher's own checksum, rather than against a digest we recorded
    ourselves, is what distinguishes "the bytes are what the publisher released" from "the bytes
    are what we happened to download".
    """
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    out = {}
    for name, record in payload.items():
        sha = record.get("sha256")
        if not sha or len(sha) != 64:
            raise ValueError(f"{name}: publisher digest missing or malformed")
        out[name] = {"sha256": sha, "bytes": int(record["bytes"])}
    return out


def verify_revision(repo_id=HF_REPO_ID, revision=HF_REVISION):
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo_id, revision=revision)
    if info.sha != revision:
        raise ValueError(f"revision mismatch: asked {revision}, got {info.sha}")
    return {"hf_repo_id": repo_id, "hf_revision": info.sha}


def fetch_heldout(target, repo_id=HF_REPO_ID, revision=HF_REVISION):
    """snapshot_download restricted to the sixteen held-out shards, at the pinned revision."""
    from huggingface_hub import snapshot_download

    target = pathlib.Path(target)
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision,
                      allow_patterns=download_patterns(), local_dir=str(target))
    return target


def resolve(name, role, local_train_dir, heldout_dir):
    return pathlib.Path(local_train_dir if role == "train" else heldout_dir) / name


def build_seal(local_train_dir, heldout_dir, publisher_digests, revision_record):
    """Fail closed on the first inconsistency. Every check names what it compared."""
    publisher = load_publisher_digests(publisher_digests)
    rows, missing, unexpected, mismatched = [], [], [], []

    expected_names = {name for _, _, name, _ in expected_files()}
    for extra in sorted(set(publisher) - expected_names):
        unexpected.append({"file": extra, "reason": "publisher digest for a non-cohort file"})

    seen_digests = {}
    for role, family, name, nominal in expected_files():
        path = resolve(name, role, local_train_dir, heldout_dir)
        if not path.is_file():
            missing.append({"file": name, "role": role, "expected_at": str(path)})
            continue
        size = int(path.stat().st_size)
        digest = sha256_file(path)
        published = publisher.get(name)
        if published is None:
            mismatched.append({"file": name, "reason": "no publisher digest at this revision"})
        else:
            if digest != published["sha256"]:
                mismatched.append({"file": name, "reason": "sha256 differs from the publisher",
                                   "local": digest, "publisher": published["sha256"]})
            if size != published["bytes"]:
                mismatched.append({"file": name, "reason": "size differs from the publisher",
                                   "local": size, "publisher": published["bytes"]})
        if size != nominal:
            mismatched.append({"file": name, "reason": "size differs from the frozen nominal",
                               "local": size, "nominal": nominal})
        if size == 0:
            mismatched.append({"file": name, "reason": "zero bytes"})
        if digest in seen_digests:
            mismatched.append({"file": name, "reason": "duplicate digest",
                               "other": seen_digests[digest]})
        seen_digests[digest] = name
        rows.append({"role": role, "family": family, "filename": name, "bytes": size,
                     "sha256": digest,
                     "source": ("preexisting_local" if role == "train"
                                else "downloaded_pinned_revision")})

    # family triples must be complete: N, N+100, N+200 all present and hashed
    by_family = {}
    for row in rows:
        by_family.setdefault(row["family"], set()).add(row["role"])
    incomplete = sorted(f for f in FAMILIES
                        if by_family.get(f, set()) != {"train", "selection", "test"})

    counts = {role: sum(1 for r in rows if r["role"] == role)
              for role in ("train", "selection", "test")}

    seal = {
        "protocol_id": PROTOCOL_ID, "stage": STAGE,
        "protocol_sha256": hashlib.sha256(
            (HERE.parent / PROTOCOL_PATH).read_bytes()).hexdigest(),
        **revision_record,
        "huggingface_hub_version": _hub_version(),
        "family_ids": list(FAMILIES),
        "files": sorted(rows, key=lambda r: (r["role"], r["family"])),
        "raw_bytes_hashed": counts,
        # hashing bytes is not opening fields. Stage A..D keep the TEST entry here at zero.
        "hdf5_field_reads": {"train": 0, "selection": 0, "test": 0},
        "missing_files": missing,
        "unexpected_files": unexpected,
        "digest_mismatches": mismatched,
        "incomplete_families": incomplete,
        "confirmation_test_metrics_computed": 0,
        "verified_against": "publisher LFS sha256 at the pinned revision",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    seal["status"] = ("DATA_SEAL_COMPLETE"
                      if not (missing or unexpected or mismatched or incomplete)
                      and counts == {"train": 8, "selection": 8, "test": 8}
                      else "DATA_SEAL_FAILED")
    return seal


def _hub_version():
    try:
        import huggingface_hub

        return huggingface_hub.__version__
    except ImportError:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True,
                        help="directory already holding the eight cohort train shards")
    parser.add_argument("--heldout-dir", required=True,
                        help="download target for the sixteen held-out shards, external volume")
    parser.add_argument("--publisher-digests", required=True,
                        help="JSON of {filename: {sha256, bytes}} from the pinned revision")
    parser.add_argument("--output", required=True, help="create-only DATA_SEAL_V1.json path")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args(argv)

    revision_record = verify_revision()
    print(f"revision verified: {revision_record['hf_revision']}", flush=True)

    if not args.skip_download:
        print(f"fetching {len(download_patterns())} held-out shards -> {args.heldout_dir}",
              flush=True)
        fetch_heldout(args.heldout_dir)

    seal = build_seal(args.train_dir, args.heldout_dir, args.publisher_digests, revision_record)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "x", encoding="utf-8") as handle:
        json.dump(seal, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps({k: seal[k] for k in
                      ("status", "raw_bytes_hashed", "hdf5_field_reads", "missing_files",
                       "unexpected_files", "digest_mismatches", "incomplete_families")},
                     indent=2, sort_keys=True))
    print("seal sha256:", sha256_file(output))
    return 0 if seal["status"] == "DATA_SEAL_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
