"""Archive raw JSON traces and per-trajectory link checks with reproducible paths."""
from pathlib import Path
import argparse
import zipfile


def package(root, destination):
    files=sorted(p for p in root.rglob("*") if p.is_file())
    with zipfile.ZipFile(destination,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for path in files:
            info=zipfile.ZipInfo(path.relative_to(root).as_posix(),(2026,11,27,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,path.read_bytes())
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
        assert len(z.infolist())==len(files)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("destination",type=Path)
    args=parser.parse_args()
    package(args.root,args.destination)
