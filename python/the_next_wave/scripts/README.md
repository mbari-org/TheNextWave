# Scripts

This folder contains helper scripts for packaging and publishing TheNextWave example-data bundles.

## Creating an example-data release

The downloadable example bundle is a tarball with this layout:

```text
example_data/
  my_example_name/
    wavespec.mat
    SWIFT22_SBG_*.mat
    SWIFT23_SBG_*.mat
    SWIFT24_SBG_*.mat
    SWIFT25_SBG_*.mat
  another_example_name/
    wavespec.mat
    SWIFT22_SBG_*.mat
    SWIFT23_SBG_*.mat
    SWIFT24_SBG_*.mat
    SWIFT25_SBG_*.mat
```

Each example folder may have any name. The Python example modes select one with
`--example-name <folder-name>`.

## Build the tarball

Run [build_example_data_release.py](build_example_data_release.py) with one or more
example folder paths:

```bash
python build_example_data_release.py /path/to/example_one /path/to/example_two
```

This writes `dist/example_data.tgz` and prints its SHA-256 hash.
Only `.mat` files are added to the tarball. Any other files in the example
folders are ignored.

To build and publish in one step:

```bash
python build_example_data_release.py \
  --publish-gh \
  --release-tag <release-tag> \
  /path/to/example_one /path/to/example_two
```

If the release already exists, the script will upload and replace the asset.
If it does not exist, the script will create the release and attach the asset.

### Script flags

[build_example_data_release.py](build_example_data_release.py) supports these arguments:

- positional `example_dirs`
  - one or more example-data folders to bundle under `example_data/`
- `--output <path>`
  - output tarball path
  - default: `dist/example_data.tgz`
- `--publish-gh`
  - after building the tarball, publish it with GitHub CLI
- `--repo <owner/repo>`
  - repository passed to `gh release ...`
  - default: `mbari-org/TheNextWave`
- `--release-tag <tag>`
  - required when using `--publish-gh`
  - the GitHub release tag to create or update
- `--release-title <title>`
  - optional release title for `gh release create`
  - default: `Example data <release-tag>`
- `--release-notes <text>`
  - optional release notes for `gh release create`
  - default: `Example data bundle for TheNextWave Python examples.`

Full example using all publish-related flags:

```bash
python build_example_data_release.py \
  --output ../dist/example_data.tgz \
  --publish-gh \
  --repo mbari-org/TheNextWave \
  --release-tag example-data-2026-05-07 \
  --release-title "Example data example-data-2026-05-07" \
  --release-notes "Adds two packaged example datasets." \
  /path/to/example_one /path/to/example_two
```

## Install GitHub CLI

GitHub releases can be created from the command line with `gh`.

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install gh
```

If `gh` is not available in your distro package list, see the official install docs:

- https://cli.github.com/

## Authenticate GitHub CLI

```bash
gh auth login
```

Choose:
- `GitHub.com`
- `HTTPS`
- authenticate in browser or paste a token

You can verify auth with:

```bash
gh auth status
```

## Create the GitHub release

Typical release steps:

1. Prepare one or more example folders, each containing `wavespec.mat` and SWIFT22–25 SBG files.
2. Run [build_example_data_release.py](build_example_data_release.py).
3. Create or choose a release tag.
4. Upload `dist/example_data.tgz` to that release.
5. Update the pinned release metadata in [python/the_next_wave/the_next_wave/download_example_data.py](../python/the_next_wave/the_next_wave/download_example_data.py):
   - `TAG`
   - `ASSET` if needed
   - `KNOWN_HASH`
6. Test the release locally.

Equivalent `gh` command if you want to run it manually:

```bash
gh release create <release-tag> dist/example_data.tgz \
  --repo mbari-org/TheNextWave \
  --title "Example data <release-tag>" \
  --notes "Example data bundle for TheNextWave Python examples."
```

If the tag already exists and you only want to upload or replace the asset:

```bash
gh release upload <release-tag> dist/example_data.tgz \
  --repo mbari-org/TheNextWave \
  --clobber
```

## Test the release locally

Use environment overrides so you can test a new release before changing the defaults in code:

```bash
export THE_NEXT_WAVE_EXAMPLE_TAG=<release-tag>
export THE_NEXT_WAVE_EXAMPLE_ASSET=example_data.tgz
export THE_NEXT_WAVE_EXAMPLE_HASH=sha256:<printed-hash>
uv run python -m the_next_wave.example --example-name <folder-name>
```