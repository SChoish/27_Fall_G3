"""Pack and restore the per-corner measurements used by audit.py."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def pack(root, path):
    with ZipFile(path, "x", ZIP_DEFLATED, compresslevel=9) as archive:
        for record in sorted(root.glob("*.json")):
            archive.write(record, "runs/" + record.name)
        archive.write(root / "summary.csv", "runs/summary.csv")
        for folder in sorted((root / "physical").iterdir()):
            rows = [p.read_bytes() for p in sorted(folder.glob("*.json"))]
            archive.writestr("physical/" + folder.name + ".jsonl", b"\n".join(rows) + b"\n")
    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive failed its CRC check")


def restore(path, root):
    root.mkdir(parents=True, exist_ok=False)
    with ZipFile(path) as archive:
        for member in archive.namelist():
            parts = Path(member).parts
            if len(parts) != 2 or parts[1] == "..":
                raise ValueError(member)
            parent, filename = parts
            if parent == "runs":
                (root / filename).write_bytes(archive.read(member))
            elif parent == "physical" and filename.endswith(".jsonl"):
                folder = root / "physical" / filename.removesuffix(".jsonl")
                folder.mkdir(parents=True)
                for line in archive.read(member).splitlines():
                    record = json.loads(line)
                    key = json.dumps([record["x"], record["corner"]]).encode()
                    (folder / (hashlib.sha256(key).hexdigest() + ".json")).write_bytes(line)
            else:
                raise ValueError(member)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("measurements.zip"))
    parser.add_argument("--restore", type=Path)
    args = parser.parse_args()
    if args.restore:
        restore(args.archive, args.restore)
    else:
        pack(Path("runs"), args.archive)
