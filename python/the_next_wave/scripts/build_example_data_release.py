#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tarfile


REQUIRED_SWIFTS = (22, 23, 24, 25)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def validate_example_dir(example_dir: Path) -> None:
    if not example_dir.is_dir():
        raise FileNotFoundError(f'Missing example directory: {example_dir}')

    wavespec = example_dir / 'wavespec.mat'
    if not wavespec.is_file():
        raise FileNotFoundError(f'Missing required file: {wavespec}')

    for swift_num in REQUIRED_SWIFTS:
        matches = sorted(example_dir.glob(f'SWIFT{swift_num}_SBG_*.mat'))
        if not matches:
            raise FileNotFoundError(
                f'Missing SWIFT{swift_num} SBG file in {example_dir}'
            )


def build_tarball(output_path: Path, examples: list[Path]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, 'w:gz') as tar:
        seen_names: set[str] = set()
        for example_dir in examples:
            validate_example_dir(example_dir)
            example_name = example_dir.name
            if example_name in seen_names:
                raise ValueError(f'Duplicate example folder name in release: {example_name}')
            seen_names.add(example_name)
            mat_files = sorted(p for p in example_dir.rglob('*.mat') if p.is_file())
            if not mat_files:
                raise FileNotFoundError(f'No .mat files found in {example_dir}')

            for mat_file in mat_files:
                rel_path = mat_file.relative_to(example_dir)
                tar.add(mat_file, arcname=f'example_data/{example_name}/{rel_path.as_posix()}')


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require_gh() -> str:
    gh = shutil.which('gh')
    if gh is None:
        raise RuntimeError(
            'GitHub CLI `gh` was not found in PATH. Install it first or omit --publish-gh.'
        )
    return gh


def release_exists(gh: str, repo: str, tag: str) -> bool:
    res = subprocess.run(
        [gh, 'release', 'view', tag, '--repo', repo],
        check=False,
        capture_output=True,
        text=True,
    )
    return res.returncode == 0


def publish_release(
    *,
    asset_path: Path,
    repo: str,
    tag: str,
    title: str,
    notes: str,
) -> None:
    gh = require_gh()

    if release_exists(gh, repo, tag):
        cmd = [
            gh,
            'release',
            'upload',
            tag,
            str(asset_path),
            '--repo',
            repo,
            '--clobber',
        ]
        print('Release exists; uploading asset with gh release upload --clobber')
    else:
        cmd = [
            gh,
            'release',
            'create',
            tag,
            str(asset_path),
            '--repo',
            repo,
            '--title',
            title,
            '--notes',
            notes,
        ]
        print('Release does not exist; creating it with gh release create')

    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Bundle ExampleData directories into a GitHub release tarball.'
    )
    p.add_argument(
        '--output',
        type=Path,
        default=repo_root() / 'dist' / 'example_data.tgz',
        help='Output tarball path.',
    )
    p.add_argument(
        'example_dirs',
        type=Path,
        nargs='+',
        help='List of example-data folders to include under example_data/ in the release.',
    )
    p.add_argument(
        '--publish-gh',
        action='store_true',
        help='Publish the built tarball to a GitHub release with `gh`.',
    )
    p.add_argument(
        '--repo',
        type=str,
        default='mbari-org/TheNextWave',
        help='GitHub repository for `gh release ...`. Default: mbari-org/TheNextWave',
    )
    p.add_argument(
        '--release-tag',
        type=str,
        default=None,
        help='Git tag / release name to create or update when using --publish-gh.',
    )
    p.add_argument(
        '--release-title',
        type=str,
        default=None,
        help='Release title for `gh release create`. Defaults to "Example data <tag>".',
    )
    p.add_argument(
        '--release-notes',
        type=str,
        default='Example data bundle for TheNextWave Python examples.',
        help='Release notes passed to `gh release create`.',
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    examples = [p.expanduser().resolve() for p in args.example_dirs]

    build_tarball(args.output.resolve(), examples)
    digest = sha256sum(args.output)
    size_mb = args.output.stat().st_size / (1024.0 * 1024.0)

    print(f'Wrote {args.output} ({size_mb:.1f} MiB)')
    print(f'sha256:{digest}')
    print()
    print('Suggested environment overrides for testing the new release:')
    print('  export THE_NEXT_WAVE_EXAMPLE_TAG=<your-release-tag>')
    print(f'  export THE_NEXT_WAVE_EXAMPLE_ASSET={args.output.name}')
    print(f'  export THE_NEXT_WAVE_EXAMPLE_HASH=sha256:{digest}')

    if args.publish_gh:
        if not args.release_tag:
            raise ValueError('--publish-gh requires --release-tag')
        release_title = args.release_title or f'Example data {args.release_tag}'
        publish_release(
            asset_path=args.output.resolve(),
            repo=str(args.repo),
            tag=str(args.release_tag),
            title=release_title,
            notes=str(args.release_notes),
        )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
