"""Extract held-out 20-lane forward responses from pinned Chiplet-SI layouts."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "implementation_results/20261125_package20/channels"
REPO = "Koyomi1999/Chiplet-SI"
REVISION = "a9408b5ed373ee76b41bb000c412cc8fd39154a7"
LANE_COUNT = 20
FREQUENCY_COUNT = 61
SEED = 26112820


def get(url):
    return subprocess.check_output(["curl","-fLsS","--max-time","25","--retry","2",url])


def raw(path):
    return get(f"https://raw.githubusercontent.com/{REPO}/{REVISION}/{path}")


def tree(revision):
    with urlopen(Request(f"https://api.github.com/repos/{REPO}/git/trees/{revision}",
                         headers={"User-Agent":"G3-channel-extraction"}),timeout=25) as response:
        data = json.load(response)
    if data["truncated"]:
        raise ValueError("GitHub returned an incomplete layout listing")
    return data["tree"]


def paths():
    sparams = next(x["sha"] for x in tree(REVISION) if x["path"]=="s_parameter_files")
    stack = next(x["sha"] for x in tree(sparams) if x["path"]=="stack_0")
    return sorted(f"s_parameter_files/stack_0/{x['path']}" for x in tree(stack)
                  if x["type"]=="blob" and x["path"].endswith("_m1.7z"))


def get_geometry(path):
    stem = Path(path).stem
    macro, layout = stem.split("_x_",1)
    return f"layouts/{macro}/x_{layout}.txt"


def central_nets(geometry):
    lines = geometry.decode().splitlines()
    label, count = lines[0].split()
    if label!="Nets" or int(count)<LANE_COUNT:
        raise ValueError("The M1 layer has fewer than 20 traces")
    nets = {}
    at = 1
    for _ in range(int(count)):
        name, n = lines[at].split()
        end = at+1+int(n)
        nets[int(name.removeprefix("Net"))] = [tuple(map(float,line.split()[1:]))
                                              for line in lines[at+1:end]]
        at = end
    start = max(min(min(a,c) for a,_,c,_ in trace) for trace in nets.values())
    end = min(max(max(a,c) for a,_,c,_ in trace) for trace in nets.values())
    if start>=end or len(nets)!=int(count):
        raise ValueError("No common cross-section across the M1 traces")
    mid = (start+end)/2

    def y_coordinate(trace):
        values = [b+(d-b)*(mid-a)/(c-a) for a,b,c,d in trace if a!=c and min(a,c)<=mid<=max(a,c)]
        if not values or not np.allclose(values,values[0],rtol=0,atol=1e-6):
            raise ValueError("A net does not have a unique midpoint crossing")
        return values[0]

    ordered = sorted(nets,key=lambda n:y_coordinate(nets[n]))
    first = (len(ordered)-LANE_COUNT)//2
    return ordered[first:first+LANE_COUNT]


def s_parameters(source):
    lines = source.decode().splitlines()
    result = {}
    if lines[0] != "Numtype 3 port id start from 0":
        raise ValueError("Unexpected S-parameter format")
    i = 1
    for _ in range(3):
        label, ports = lines[i].split()
        if label!="Nump1":
            raise ValueError("Missing S-parameter section")
        i += 1
        for _ in range(int(ports)):
            a, pairs = lines[i].split()
            i += 1
            for _ in range(int(pairs)):
                b, length = lines[i].split()
                v = np.fromstring(lines[i+1],sep=" ")
                key = int(a[1:]),int(b[1:])
                if len(v)!=2*FREQUENCY_COUNT or len(v)!=int(length) or key in result:
                    raise ValueError("Malformed or duplicate S-parameter record")
                result[key] = np.power(10,v[:FREQUENCY_COUNT]/20)*np.exp(1j*np.deg2rad(v[FREQUENCY_COUNT:]))
                i += 2
    if i!=len(lines):
        raise ValueError("Unexpected trailing S-parameter data")
    return result


def extract(path):
    archive = raw(path)
    geom_path = get_geometry(path)
    geom = raw(geom_path)
    # Only read the archive's text stream; never unpack paths into the workspace.
    with NamedTemporaryFile(suffix=".7z") as downloaded:
        downloaded.write(archive)
        downloaded.flush()
        source = subprocess.check_output(["bsdtar","-xOf",downloaded.name])
    nets = central_nets(geom)
    response = s_parameters(source)
    zero = np.zeros(FREQUENCY_COUNT,complex)
    h = np.array([[response.get((2*rx,2*tx+1),zero) for tx in nets] for rx in nets])
    active = np.any(abs(h)>0,axis=2)
    if h.shape!=(20,20,61) or not np.all(active.diagonal()) or active.sum()<=20 or not np.isfinite(h).all():
        raise ValueError("Missing through paths, crosstalk, or valid 20-lane samples")
    details = dict(path=path, geometry=geom_path, sha256_archive=sha256(archive).hexdigest(),
                   sha256_source=sha256(source).hexdigest(), sha256_geometry=sha256(geom).hexdigest(),
                   nets=nets, through_paths=20, crosstalk_paths=int(active.sum()-20),
                   matrix_sha256=sha256(h.tobytes()).hexdigest())
    return h,details


def sample(items):
    grouped = defaultdict(list)
    for p in items:
        grouped[Path(p).stem.split("_x_",1)[0]].append(p)
    rng = np.random.default_rng(SEED)
    macros = rng.permutation(sorted(grouped))
    queues = [list(rng.permutation(grouped[m])) for m in macros]
    return [str(group[i]) for i in range(max(map(len,queues))) for group in queues if i<len(group)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    refs = [json.loads((REFERENCE/f"layout{i}.json").read_text()) for i in range(3)]
    with ThreadPoolExecutor(8) as pool:
        for i,(h,a) in enumerate(pool.map(extract,[r["path"] for r in refs])):
            with np.load(REFERENCE/f"layout{i}.npz") as baseline:
                np.testing.assert_array_equal(h,baseline["h"])
                np.testing.assert_array_equal(a["nets"],baseline["nets"])
        ordered = sample([p for p in paths() if p not in {r["path"] for r in refs}])
        if len(ordered)<50:
            raise ValueError("Fewer than 50 new M1 layouts available")
        matrices, rows, rejections = [], [], []

        def attempt(path):
            try:
                return extract(path)
            except ValueError as error:
                return path,str(error)

        for start in range(0,len(ordered),10):
            for h,row in pool.map(attempt,ordered[start:start+10]):
                if isinstance(h,str):
                    rejections.append(dict(path=h,reason=row))
                elif row["matrix_sha256"] in {r["matrix_sha256"] for r in rows}:
                    rejections.append(dict(path=row["path"],reason="duplicate forward response"))
                else:
                    matrices.append(h)
                    rows.append(dict(layout=len(rows),**row))
                if len(rows)==50:
                    break
            if len(rows)==50:
                break
    if len(rows)!=50:
        raise ValueError("Failed to get 50 distinct physical layouts")
    np.savez_compressed(args.out/"channels.npz",h=np.stack(matrices))
    (args.out/"layouts.json").write_text(json.dumps(dict(repo=REPO,commit=REVISION,license="CC BY 4.0",
        selection_seed=SEED,reference_matrices_reproduced=3,layouts=rows,rejections=rejections),indent=2)+"\n")
    print("Exactly re-created all 3 reference matrices; extracted 50 other original 20-lane layouts.",flush=True)


if __name__=="__main__":
    main()
