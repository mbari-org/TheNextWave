from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pooch

from .swift import WaveSpec, loadmat_struct
from .utilities import build_wavespec_from_directional_spectrum, select_sbg_burst_struct


OWNER = 'mbari-org'
REPO = 'TheNextWave'
PKG = 'the_next_wave'
TAG = 'example'  # pin a tag for determinism (avoid 'latest')
ASSET = 'example_data.tgz'
URL = f'https://github.com/{OWNER}/{REPO}/releases/download/{TAG}/{ASSET}'
KNOWN_HASH = 'sha256:90aab5a12d6b99a87a342b8e20d0367bde858bb174c5a79fdea9352594f18d8b'

ENV_OVERRIDE = 'THE_NEXT_WAVE_EXAMPLE_DATA_DIR'
ENV_TAG = 'THE_NEXT_WAVE_EXAMPLE_TAG'
ENV_ASSET = 'THE_NEXT_WAVE_EXAMPLE_ASSET'
ENV_HASH = 'THE_NEXT_WAVE_EXAMPLE_HASH'
ENV_URL = 'THE_NEXT_WAVE_EXAMPLE_URL'

DEFAULT_EXAMPLE_NUM = 1
VALID_EXAMPLE_NUMS = (1, 2)
EXAMPLE_SELECT_IDX = {
    1: 91,  # MATLAB burst index 92
    2: 9,   # MATLAB burst index 10
}
LEGACY_SELECT_IDX_BY_NAME = {
    'ExampleData1': 91,
    'ExampleData2': 9,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalize_data_root(path: Path) -> Path | None:
    if not path.is_dir():
        return None

    if (path / 'ExampleData1').is_dir() or (path / 'ExampleData2').is_dir():
        return path

    if (path / 'example_data').is_dir():
        return path / 'example_data'

    if path.name.startswith('ExampleData') and (path / 'wavespec.mat').is_file():
        return path.parent

    if (path / 'wavespec.mat').is_file():
        return path

    return None


def _looks_like_example_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / 'wavespec.mat').is_file():
        return False

    for swift_num in range(22, 26):
        if not any(path.glob(f'SWIFT{swift_num}_SBG_*.mat')):
            return False
    return True


def _validate_example_num(example_num: int) -> int:
    example_num = int(example_num)
    if example_num not in VALID_EXAMPLE_NUMS:
        raise ValueError(
            f'Invalid example_num={example_num}; expected one of {VALID_EXAMPLE_NUMS}'
        )
    return example_num


def _resolve_legacy_name(example_num: int) -> str:
    example_num = _validate_example_num(example_num)
    return f'ExampleData{example_num}'


def _downloaded_data_root() -> Path:
    tag = os.environ.get(ENV_TAG, TAG)
    asset = os.environ.get(ENV_ASSET, ASSET)
    known_hash = os.environ.get(ENV_HASH, KNOWN_HASH)
    url = os.environ.get(ENV_URL, URL.replace(TAG, tag).replace(ASSET, asset))

    cache_root = pooch.os_cache(PKG) / tag
    cache_root.mkdir(parents=True, exist_ok=True)

    extract_dir = cache_root / 'extracted'
    pooch.retrieve(
        url=url,
        fname=asset,
        path=str(cache_root),
        known_hash=known_hash if known_hash else None,
        processor=pooch.Untar(extract_dir=str(extract_dir)),
        progressbar=True,
    )

    root = _normalize_data_root(extract_dir)
    if root is None:
        raise FileNotFoundError(
            f'Could not locate extracted example data under {extract_dir}'
        )
    return root


def get_example_data_root() -> Path:
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        p = Path(override).expanduser().resolve()
        root = _normalize_data_root(p)
        if root is not None:
            return root
        raise FileNotFoundError(
            f'{ENV_OVERRIDE} points to {p}, but no example dataset was found there'
        )

    repo_root = _repo_root()
    root = _normalize_data_root(repo_root)
    if root is not None:
        return root

    return _downloaded_data_root()


def list_example_data_dirs() -> dict[str, Path]:
    root = get_example_data_root()

    out: dict[str, Path] = {}
    for child in sorted(root.iterdir()):
        if _looks_like_example_dir(child):
            out[child.name] = child

    if out:
        return out

    # Backward compatibility with the original flat bundle layout.
    if _looks_like_example_dir(root):
        out[root.name] = root

    return out


def get_default_example_name() -> str:
    examples = list_example_data_dirs()
    if not examples:
        raise FileNotFoundError(
            f'No example datasets found under {get_example_data_root()}'
        )

    if 'ExampleData1' in examples:
        return 'ExampleData1'

    return sorted(examples)[0]


def get_example_data_dir(
    example_name: str | None = None,
    example_num: int | None = None,
) -> Path:
    if example_name is not None and example_num is not None:
        raise ValueError('Pass only one of example_name or example_num')

    if example_num is not None:
        example_name = _resolve_legacy_name(example_num)

    if example_name is None:
        example_name = get_default_example_name()

    examples = list_example_data_dirs()

    candidate = examples.get(str(example_name))
    if candidate is not None:
        return candidate

    raise FileNotFoundError(
        f'Example dataset {example_name!r} not found under {get_example_data_root()}. '
        f'Available examples: {sorted(examples)}'
    )


def get_example_select_idx(
    example_name: str | None = None,
    example_num: int | None = None,
) -> int:
    if example_name is not None and example_num is not None:
        raise ValueError('Pass only one of example_name or example_num')

    if example_num is not None:
        return int(EXAMPLE_SELECT_IDX[_validate_example_num(example_num)])

    if example_name is None:
        example_name = get_default_example_name()

    if example_name in LEGACY_SELECT_IDX_BY_NAME:
        return int(LEGACY_SELECT_IDX_BY_NAME[example_name])

    raise KeyError(
        f'No legacy SWIFT select index is defined for example dataset {example_name!r}'
    )


def get_example_sbg_paths(
    example_name: str | None = None,
    example_num: int | None = None,
) -> tuple[Path, Path, Path, Path]:
    example_dir = get_example_data_dir(example_name=example_name, example_num=example_num)
    out = []
    for swift_num in range(22, 26):
        matches = sorted(example_dir.glob(f'SWIFT{swift_num}_SBG_*.mat'))
        if not matches:
            raise FileNotFoundError(
                f'No SWIFT{swift_num} SBG .mat file found in {example_dir}'
            )
        out.append(matches[0])
    return tuple(out)


def get_example_swift_paths(
    example_name: str | None = None,
    example_num: int | None = None,
) -> tuple[Path, Path, Path, Path] | None:
    example_dir = get_example_data_dir(example_name=example_name, example_num=example_num)
    out = []
    for swift_num in range(22, 26):
        matches = sorted(
            example_dir.glob(
                f'SWIFT{swift_num}_DIGIFLOAT_*_reprocessedSBG_displacements.mat'
            )
        )
        if not matches:
            return None
        out.append(matches[0])
    return tuple(out)


def load_example_sbg_bursts(
    example_name: str | None = None,
    example_num: int | None = None,
):
    bursts = []
    for path in get_example_sbg_paths(example_name=example_name, example_num=example_num):
        mat = loadmat_struct(path)
        sbg_data = mat['sbgData']
        bursts.append(select_sbg_burst_struct(sbg_data, prefer_longest=True))
    return tuple(bursts)


def load_example_wavespec(
    example_name: str | None = None,
    example_num: int | None = None,
) -> WaveSpec:
    mat_path = get_example_data_dir(example_name=example_name, example_num=example_num) / 'wavespec.mat'
    if not mat_path.is_file():
        raise FileNotFoundError(f'wavespec.mat not found in {mat_path.parent}')

    mat = loadmat_struct(mat_path)
    ws_in = mat['wavespec']
    return build_wavespec_from_directional_spectrum(ws_in.Etheta, ws_in.theta, ws_in.f)
