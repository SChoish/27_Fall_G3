"""Reject artifacts produced for different circuit, measurement or task contracts.

New artifacts have a hash-bound sidecar. Original release artifacts are accepted
only against the pinned release manifest AND the frozen legacy contract.
"""
import hashlib
import inspect
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parent


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract(cfg):
    from coupled_experiment import packet_for, features, quality, utility, rank, Environment, Table
    return {
        'schema': 1,
        'config_sha256': hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'model_sha256': file_hash(ROOT / 'assets/45nm_bulk.txt'),
        'measurement_code': {name: file_hash(ROOT / name) for name in ('link.py', 'coupled_link.py')},
        'task_code': {obj.__qualname__: hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()
                      for obj in (packet_for, features, quality, utility, rank, Environment, Table.evaluate)},
    }


def metadata_path(path):
    return Path(str(path) + '.metadata.json')


def write_metadata(path, cfg, dependencies=()):
    path = Path(path)
    payload = {'contract': contract(cfg), 'artifact_sha256': file_hash(path),
               'dependencies': {str(Path(p).resolve().relative_to(ROOT)): file_hash(p) for p in dependencies}}
    destination = metadata_path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=destination.parent, delete=False) as f:
        temporary = Path(f.name)
        json.dump(payload, f, indent=2)
        f.write('\n')
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def verify_artifact(path, cfg, dependencies=()):
    path = Path(path)
    current = contract(cfg)
    sidecar = metadata_path(path)
    if sidecar.exists():
        payload = json.loads(sidecar.read_text())
        if payload.get('contract') != current:
            raise ValueError(f'Stale artifact contract: {path}. Regenerate for the current configuration/code.')
        if payload.get('artifact_sha256') != file_hash(path):
            raise ValueError(f'Artifact checksum mismatch: {path}')
        recorded = payload.get('dependencies', {})
        required = {str(Path(p).resolve().relative_to(ROOT)) for p in dependencies}
        if not required <= recorded.keys():
            raise ValueError(f'Missing artifact dependencies: {path}')
        for name, digest in recorded.items():
            dependency = (ROOT / name).resolve()
            if not dependency.is_relative_to(ROOT) or file_hash(dependency) != digest:
                raise ValueError(f'Artifact dependency mismatch: {name}')
        return
    # Never infer provenance from the current file contents or silently bless
    # an unrecognized old artifact. This inventory is from the original release.
    legacy = json.loads((ROOT / 'legacy_contract.json').read_text())
    if legacy['contract'] != current:
        raise ValueError(f'Stale legacy artifact contract: {path}. Regenerate for the current configuration/code.')
    manifest = json.loads((ROOT / 'COUPLED_MANIFEST_SHA256.json').read_text())
    if file_hash(ROOT / 'COUPLED_MANIFEST_SHA256.json') != legacy['manifest_sha256']:
        raise ValueError('Original release manifest checksum mismatch')
    for candidate in (path, *map(Path, dependencies)):
        try:
            name = str(candidate.resolve().relative_to(ROOT))
        except ValueError:
            raise ValueError(f'Artifact has no provenance: {candidate}') from None
        if name not in manifest or manifest[name]['sha256'] != file_hash(candidate):
            raise ValueError(f'Artifact has no matching release provenance: {candidate}')
