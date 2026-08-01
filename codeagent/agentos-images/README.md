# Agent OS openEuler x86 Image Build

Build and export **amd64** images for Agent OS:

- `agent-base:1.0-<oe-tag>-amd64`
- `agentos-image-process:<oe-tag>-amd64`

Default matrix: **openEuler 24.03-lts-sp1 ~ sp4** only (OE22 skipped: system Python 3.9 &lt; required 3.11).

Source Dockerfiles live in the `agent-os` repo:

- `control-panel/image_process/base.Dockerfile`
- `control-panel/image_process/Dockerfile`

## Prerequisites

- WSL / Linux with Docker (Docker Desktop + WSL works)
- Checkout of `agent-os` with the parameterized Dockerfiles above
- Network access to:
  - Docker Hub (`openeuler/openeuler`)
  - Huawei Node mirror
  - openYuanrong OBS daily index

## Layout (example)

| Item | Path |
|------|------|
| agent-os repo | `/home/csoulnd/project_os/agent-os` |
| These scripts | this directory (or copy to a working path) |
| Output | `/mnt/c/data/images/20260801_openeuler_x86/<oe-tag>/` |

Each version directory contains:

```text
agent-base_1.0_<oe-tag>_amd64.tar
agentos-image-process_<oe-tag>_amd64.tar
MANIFEST.txt
```

## Quick start (x86)

```bash
cd codeagent/agentos-images
chmod +x build-oe-matrix.sh

# Smoke: only 24.03-lts-sp4
./build-oe-matrix.sh \
  --arch amd64 \
  --series 24 \
  --sps 4 \
  --repo-root /home/csoulnd/project_os/agent-os \
  --out-root /mnt/c/data/images/20260801_openeuler_x86

# Full OE24 SP1-SP4
./build-oe-matrix.sh \
  --arch amd64 \
  --series 24 \
  --sps 1,2,3,4 \
  --repo-root /home/csoulnd/project_os/agent-os \
  --out-root /mnt/c/data/images/20260801_openeuler_x86
```

Defaults if omitted: `--arch` = host arch, `--series 24`, `--sps 1,2,3,4`, yuanrong schedule time = latest from OBS daily index.

## Important notes

1. **Docker Desktop + WSL**: do not use `docker save -o /mnt/c/...`. The script redirects stdout into the tar path so files land on the Windows mount.
2. **Node arch naming**: Node uses `linux-x64` (not `linux-amd64`); handled inside `base.Dockerfile`.
3. **yuanrong SDK**: schedule time is auto-resolved; override with `--yr-schedule-time` if needed.
4. **Deploy tag bridge** after `docker load`:

```bash
docker tag agent-base:1.0-24.03-lts-sp4-amd64 agent-base:1.0
docker tag agentos-image-process:24.03-lts-sp4-amd64 agentos-image-process:latest
```

## Script options

| Option | Meaning |
|--------|---------|
| `--out-root DIR` | Artifact root |
| `--repo-root DIR` | agent-os checkout |
| `--series LIST` | e.g. `24` |
| `--sps LIST` | e.g. `1,2,3,4` |
| `--arch amd64` | Force x86_64 |
| `--yr-schedule-time TS` | Pin OBS daily build number |
| `--yuanrong-version VER` | Wheel version (default `9.9.9`) |
