#!/usr/bin/env bash
# Build agent-base + agentos-image-process for openEuler (x86/amd64).
# Docs: README.md in this directory.
#
# Smoke:
#   ./build-oe-matrix.sh --arch amd64 --series 24 --sps 4 \
#     --repo-root /home/csoulnd/project_os/agent-os \
#     --out-root /mnt/c/data/images/20260801_openeuler_x86
#
# Full OE24 SP1-SP4:
#   ./build-oe-matrix.sh --arch amd64 --series 24 --sps 1,2,3,4 \
#     --repo-root /home/csoulnd/project_os/agent-os \
#     --out-root /mnt/c/data/images/20260801_openeuler_x86
#
# Bridge to deploy defaults after docker load:
#   docker tag agent-base:1.0-<oe-tag>-amd64 agent-base:1.0
#   docker tag agentos-image-process:<oe-tag>-amd64 agentos-image-process:latest

set -euo pipefail

# Empty = resolve latest from OBS daily index.
YR_SCHEDULE_TIME="${YR_SCHEDULE_TIME:-}"
YUANRONG_VERSION="${YUANRONG_VERSION:-9.9.9}"
SERIES="${SERIES:-24}"
SPS="${SPS:-1,2,3,4}"
REPO_ROOT="${REPO_ROOT:-/home/csoulnd/project_os/agent-os}"
YUANRONG_DAILY_INDEX_URL="${YUANRONG_DAILY_INDEX_URL:-https://openyuanrong.obs.cn-southwest-2.myhuaweicloud.com/daily_build/index.html}"

HOST_M="$(uname -m)"
case "$HOST_M" in
  aarch64|arm64) DEFAULT_ARCH=arm64 ;;
  *) DEFAULT_ARCH=amd64 ;;
esac
ARCH="${ARCH:-$DEFAULT_ARCH}"

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \?//'
  cat <<EOF

Options:
  --out-root DIR          Output root
  --yr-schedule-time TS   OBS daily build number (default: auto from index)
  --yuanrong-version VER  Wheel version (default: $YUANRONG_VERSION)
  --series LIST           Comma-separated majors (default: 24)
  --sps LIST              Comma-separated SP numbers (default: 1,2,3,4)
  --arch ARCH             amd64 (recommended for this doc) or arm64
  --repo-root DIR         agent-os repo root (default: $REPO_ROOT)
  -h, --help              Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --yr-schedule-time) YR_SCHEDULE_TIME="$2"; shift 2 ;;
    --yuanrong-version) YUANRONG_VERSION="$2"; shift 2 ;;
    --series) SERIES="$2"; shift 2 ;;
    --sps) SPS="$2"; shift 2 ;;
    --arch) ARCH="$2"; shift 2 ;;
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

case "$ARCH" in
  amd64|x86_64) ARCH=amd64; ARCH_LONG=x86_64; ARCH_DIR_SUFFIX=x86 ;;
  arm64|aarch64) ARCH=arm64; ARCH_LONG=aarch64; ARCH_DIR_SUFFIX=arm ;;
  *) echo "unsupported arch: $ARCH" >&2; exit 1 ;;
esac

OUT_ROOT="${OUT_ROOT:-/mnt/c/data/images/20260801_openeuler_${ARCH_DIR_SUFFIX}}"
PLATFORM="linux/$ARCH"

fetch_latest_yr_schedule_time() {
  local html schedule_time
  echo "  fetch latest openeuler build from: ${YUANRONG_DAILY_INDEX_URL}" >&2
  html="$(curl -fsSL --retry 3 --retry-delay 2 --connect-timeout 30 --max-time 60 \
    "${YUANRONG_DAILY_INDEX_URL}")"
  schedule_time="$(printf '%s\n' "${html}" \
    | sed -n '/<h2>openeuler<\/h2>/,/<hr class="os-divider">/p' \
    | sed -n 's/.*<tr><td>\([0-9][0-9]*\)<\/td>.*/\1/p' \
    | head -1)"
  if [[ -z "${schedule_time}" ]]; then
    schedule_time="$(printf '%s\n' "${html}" \
      | grep -oE 'daily_build/[0-9]+/openeuler' \
      | head -1 \
      | sed -E 's|daily_build/([0-9]+)/openeuler|\1|')"
  fi
  if [[ -z "${schedule_time}" ]]; then
    echo "error: failed to resolve latest yuanrong daily build" >&2
    return 1
  fi
  echo "${schedule_time}"
}

if [[ -z "${YR_SCHEDULE_TIME}" ]]; then
  YR_SCHEDULE_TIME="$(fetch_latest_yr_schedule_time)"
fi
SDK_BASE_URL="https://openyuanrong.obs.cn-southwest-2.myhuaweicloud.com/daily_build/${YR_SCHEDULE_TIME}/openeuler/${ARCH_LONG}"

CONTEXT="$REPO_ROOT/control-panel/image_process"
BASE_DOCKERFILE="$CONTEXT/base.Dockerfile"
SERVICE_DOCKERFILE="$CONTEXT/Dockerfile"

[[ -f "$BASE_DOCKERFILE" ]] || { echo "base.Dockerfile not found: $BASE_DOCKERFILE" >&2; exit 1; }
[[ -f "$SERVICE_DOCKERFILE" ]] || { echo "Dockerfile not found: $SERVICE_DOCKERFILE" >&2; exit 1; }

IFS=',' read -r -a SERIES_ARR <<< "$SERIES"
IFS=',' read -r -a SPS_ARR <<< "$SPS"

OE_TAGS=()
for maj in "${SERIES_ARR[@]}"; do
  maj="${maj// /}"
  [[ -n "$maj" ]] || continue
  case "$maj" in
    24) prefix="24.03-lts-sp" ;;
    22) prefix="22.03-lts-sp" ;;
    *) echo "unsupported series '$maj' (use 24 or 22)" >&2; exit 1 ;;
  esac
  for sp in "${SPS_ARR[@]}"; do
    sp="${sp// /}"
    [[ -n "$sp" ]] || continue
    OE_TAGS+=("${prefix}${sp}")
  done
done

echo "RepoRoot    : $REPO_ROOT"
echo "OutRoot     : $OUT_ROOT"
echo "Platform    : $PLATFORM"
echo "SDK         : $SDK_BASE_URL"
echo "OE tags     : ${OE_TAGS[*]}"
echo "Context     : $CONTEXT"
mkdir -p "$OUT_ROOT"

run_docker() {
  echo ">> docker $*"
  docker "$@"
}

for oe in "${OE_TAGS[@]}"; do
  base_os="openeuler/openeuler:${oe}"
  out_dir="$OUT_ROOT/$oe"
  mkdir -p "$out_dir"

  base_tag="agent-base:1.0-${oe}-${ARCH}"
  svc_tag="agentos-image-process:${oe}-${ARCH}"
  base_tar="$out_dir/agent-base_1.0_${oe}_${ARCH}.tar"
  svc_tar="$out_dir/agentos-image-process_${oe}_${ARCH}.tar"

  echo ""
  echo "======== Building ${oe} (${PLATFORM}) ========"

  run_docker pull --platform "$PLATFORM" "$base_os"

  run_docker build --platform "$PLATFORM" \
    -f "$BASE_DOCKERFILE" \
    -t "$base_tag" \
    --build-arg "BASE_OS_IMAGE=$base_os" \
    --build-arg "YR_SCHEDULE_TIME=$YR_SCHEDULE_TIME" \
    --build-arg "YUANRONG_VERSION=$YUANRONG_VERSION" \
    --build-arg "ARCH=$ARCH" \
    --build-arg "ARCH_LONG=$ARCH_LONG" \
    --build-arg "SDK_BASE_URL=$SDK_BASE_URL" \
    "$CONTEXT"

  run_docker build --platform "$PLATFORM" \
    -f "$SERVICE_DOCKERFILE" \
    -t "$svc_tag" \
    --build-arg "BASE_OS_IMAGE=$base_os" \
    "$CONTEXT"

  # Shell redirect: Docker Desktop daemon cannot write -o into WSL /mnt/c paths.
  rm -f "$base_tar" "$svc_tar"
  echo ">> docker save ${base_tag} > ${base_tar}"
  docker save "$base_tag" > "$base_tar"
  echo ">> docker save ${svc_tag} > ${svc_tar}"
  docker save "$svc_tag" > "$svc_tar"
  [[ -s "$base_tar" ]] || { echo "error: empty tar $base_tar" >&2; exit 1; }
  [[ -s "$svc_tar" ]] || { echo "error: empty tar $svc_tar" >&2; exit 1; }

  base_id="$(docker image inspect -f '{{.Id}}' "$base_tag")"
  svc_id="$(docker image inspect -f '{{.Id}}' "$svc_tag")"
  built_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  cat > "$out_dir/MANIFEST.txt" <<EOF
built_at=$built_at
oe_tag=$oe
arch=$ARCH
arch_long=$ARCH_LONG
platform=$PLATFORM
yr_schedule_time=$YR_SCHEDULE_TIME
yuanrong_version=$YUANRONG_VERSION
sdk_base_url=$SDK_BASE_URL
base_os_image=$base_os
agent_base_image=$base_tag
agent_base_id=$base_id
agent_base_tar=$(basename "$base_tar")
image_process_image=$svc_tag
image_process_id=$svc_id
image_process_tar=$(basename "$svc_tar")

# Bridge to deploy defaults after docker load:
#   docker tag $base_tag agent-base:1.0
#   docker tag $svc_tag agentos-image-process:latest
EOF

  echo "Saved -> $out_dir"
done

echo ""
echo "Done. Artifacts under $OUT_ROOT"
echo "Deploy tag bridge example:"
echo "  docker tag agent-base:1.0-24.03-lts-sp4-${ARCH} agent-base:1.0"
echo "  docker tag agentos-image-process:24.03-lts-sp4-${ARCH} agentos-image-process:latest"
