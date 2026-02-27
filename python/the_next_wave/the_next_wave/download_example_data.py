import os
from pathlib import Path

import pooch


OWNER = 'mbari-org'
REPO = 'TheNextWave'
PKG = 'the_next_wave'
TAG = 'example'  # pin a tag for determinism (avoid 'latest')
ASSET = 'example_data.tgz'
URL = f'https://github.com/{OWNER}/{REPO}/releases/download/{TAG}/{ASSET}'
KNOWN_HASH = 'sha256:5ee1faa4139b9e54b96bd50320afa38194c2f68358791a7b2f90ea862de991be'
ENV_OVERRIDE = ''


def get_example_data_dir() -> Path:
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        p = Path(override).expanduser().resolve()
        if p.is_dir():
            return p
        raise FileNotFoundError(f'{ENV_OVERRIDE} points to missing dir: {p}')

    cache_root = pooch.os_cache(PKG) / TAG
    cache_root.mkdir(parents=True, exist_ok=True)

    extract_dir = cache_root / 'extracted'
    pooch.retrieve(
        url=URL,
        fname=ASSET,
        path=str(cache_root),
        known_hash=KNOWN_HASH,
        processor=pooch.Untar(extract_dir=str(extract_dir)),
        progressbar=True,
    )

    # tarball contains a top-level 'example_data/' folder:
    candidate = extract_dir / 'example_data'
    return candidate if candidate.is_dir() else extract_dir
