#!/usr/bin/env bash
#
# Run TheNextWave against the MBARI WEC Gazebo sim twice -- once with
# single-direction (unspread) incident waves and once with directional
# spreading -- recording a separate rosbag per case, then export CSVs.
#
# Produces, under $OUT_DIR (default: ./sim_runs/<timestamp>):
#   nospread/bag/            rosbag2 of /latent_data + /wave_predictions
#   nospread_input.csv       simulator incident-wave truth at target + SWIFTs
#   nospread_output.csv      predictor output (one row per predicted sample)
#   spread/bag/              "
#   spread_input.csv         "
#   spread_output.csv        "
#
# Usage:
#   scripts/run_sim_cases.sh [-d SIM_SECONDS] [-o OUT_DIR] [-c CASES] [-H] [-x]
#
#   -d  simulated seconds to record per case          (default: 600)
#   -o  output directory                              (default: ./sim_runs/<timestamp>)
#   -c  comma-separated cases to run                  (default: nospread,spread)
#   -H  run Gazebo with a GUI (default is headless)
#   -x  also export dense + spectrum CSVs
#
# Environment: source /opt/ros/<distro>/setup.bash and the workspace
# install/setup.bash. Do NOT overwrite PYTHONPATH afterwards -- the predictor
# node resolves its console-script metadata from
# install/the_next_wave/lib/python3.12/site-packages, and clobbering PYTHONPATH
# makes the node die at startup while Gazebo keeps running (so you get a bag
# with /latent_data but no /wave_predictions). Append to PYTHONPATH instead:
#
#   source /opt/ros/jazzy/setup.bash
#   source install/setup.bash
#   export PYTHONPATH="$PYTHONPATH:$PWD/install/gz_sim_vendor/opt/gz_sim_vendor/lib/python:$PWD/install/gz_math_vendor/opt/gz_math_vendor/lib/python"
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"

DURATION_SEC=600
OUT_DIR=""
CASES="nospread,spread"
HEADLESS=true
EXTRA_CSV=false

while getopts ':d:o:c:Hxh' opt; do
  case "$opt" in
    d) DURATION_SEC="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    c) CASES="$OPTARG" ;;
    H) HEADLESS=false ;;
    x) EXTRA_CSV=true ;;
    h) sed -n '2,34p' "${BASH_SOURCE[0]}" | sed 's/^#\s\?//'; exit 0 ;;
    *) echo "unknown option -$OPTARG" >&2; exit 64 ;;
  esac
done

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="${PKG_DIR}/sim_runs/$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

if [[ -z "${ROS_DISTRO:-}" ]]; then
  echo "ERROR: no ROS 2 environment sourced (source /opt/ros/<distro>/setup.bash and your workspace install/setup.bash first)" >&2
  exit 1
fi

# Preflight: a broken PYTHONPATH kills the predictor node seconds after launch
# while Gazebo happily runs to completion, which silently yields a bag with no
# /wave_predictions. Fail loudly here instead, before burning a full run.
preflight_env() {
  local failed=0

  # Console-script metadata for the predictor node itself.
  if ! python3 -c "import importlib.metadata as m; m.version('the-next-wave')" 2>/dev/null; then
    echo "ERROR: python dist 'the-next-wave' not found." >&2
    echo "       Source the workspace install/setup.bash and do not overwrite PYTHONPATH afterwards" >&2
    echo "       (or 'pip install --user .' from ${PKG_DIR})." >&2
    failed=1
  fi

  # Message packages needed by the node, the recorder, and bag_to_csv.py.
  local mod
  for mod in buoy_interfaces.msg buoy_api.interface the_next_wave rosbag2_py; do
    if ! python3 -c "import ${mod}" 2>/dev/null; then
      echo "ERROR: cannot import python module '${mod}' -- check your sourced environment." >&2
      failed=1
    fi
  done

  # The simulator launch we include.
  if ! ros2 pkg prefix buoy_gazebo >/dev/null 2>&1; then
    echo "ERROR: ros package 'buoy_gazebo' not found on AMENT_PREFIX_PATH." >&2
    failed=1
  fi

  if (( failed )); then
    echo >&2
    echo "Preflight failed; refusing to start. See the 'Environment' notes at the top of this script." >&2
    exit 1
  fi
  echo " preflight  : ok (the-next-wave, buoy_interfaces, buoy_api, rosbag2_py, buoy_gazebo)"
}

# Sim runs are localhost-only; keep DDS traffic off the LAN and out of other runs.
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"

INPUT_TOPIC=/latent_data
OUTPUT_TOPIC=/wave_predictions

echo "=============================================================="
echo " output dir : $OUT_DIR"
echo " cases      : $CASES"
echo " duration   : ${DURATION_SEC} sim seconds per case"
echo " headless   : $HEADLESS"
preflight_env
echo "=============================================================="

# Wait up to $2 seconds for every process in group $1 to exit.
wait_pgid_gone() {
  local pgid="$1" grace="$2" waited=0
  while kill -0 "-${pgid}" 2>/dev/null && (( waited < grace )); do
    sleep 1
    waited=$((waited + 1))
  done
  ! kill -0 "-${pgid}" 2>/dev/null
}

# Shut down a process group: INT, then TERM, then KILL as a last resort.
#
# `ros2 bag record` ignores SIGINT when stdin is not a tty (it reserves INT for
# its keyboard handler), but shuts down cleanly on SIGTERM -- flushing the cache
# and finalizing the mcap. So keep the INT grace short and give TERM the long
# grace, and only KILL if TERM is also ignored.
stop_pgid() {
  local pgid="$1" name="$2" int_grace="${3:-5}" term_grace="${4:-60}"
  [[ -z "$pgid" ]] && return 0
  kill -0 "-${pgid}" 2>/dev/null || return 0

  kill -INT "-${pgid}" 2>/dev/null
  wait_pgid_gone "$pgid" "$int_grace" && return 0

  echo "  $name still up after ${int_grace}s; sending TERM (allowing ${term_grace}s to flush)"
  kill -TERM "-${pgid}" 2>/dev/null
  wait_pgid_gone "$pgid" "$term_grace" && return 0

  echo "  WARNING: $name ignored TERM after ${term_grace}s; sending KILL (output may be truncated)" >&2
  kill -KILL "-${pgid}" 2>/dev/null
  wait_pgid_gone "$pgid" 5
  return 0
}

# Process group of $1, as created by `setsid`. Falls back to the pid itself,
# which is correct because setsid makes the child its own group leader.
#
# Guards against returning this script's own process group: if setsid somehow
# failed, the child would share our group and `kill -- -PGID` would take out the
# whole run. In that case return nothing so stop_pgid becomes a no-op.
OWN_PGID="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"
pgid_of() {
  local pid="$1" pgid
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
  pgid="${pgid:-$pid}"
  if [[ -n "$OWN_PGID" && "$pgid" == "$OWN_PGID" ]]; then
    echo "  WARNING: child $pid shares this script's process group; not tracking it for shutdown" >&2
    return 0
  fi
  echo "$pgid"
}

LAUNCH_PGID=""
BAG_PGID=""
cleanup() {
  stop_pgid "$BAG_PGID" "bag recorder" 5 60
  stop_pgid "$LAUNCH_PGID" "launch" 20 60
}
trap 'echo; echo "interrupted -- cleaning up"; cleanup; exit 130' INT TERM

run_case() {
  local case_name="$1"
  local config="${PKG_DIR}/config/config_sim_${case_name}.yaml"
  local case_dir="${OUT_DIR}/${case_name}"
  local bag_dir="${case_dir}/bag"

  if [[ ! -f "$config" ]]; then
    echo "ERROR: missing config $config" >&2
    return 1
  fi
  mkdir -p "$case_dir"
  rm -rf "$bag_dir"

  echo
  echo "--- case '${case_name}' -----------------------------------------"
  echo "  config : $config"
  grep -E '^\s*(wave_dir|Hs|Tp|n_phases|spreading_deg):' "$config" | sed 's/^/    /'

  setsid ros2 launch the_next_wave the_next_wave.launch.py \
      params_file:="$config" \
      gzsim_headless:="$HEADLESS" \
      gzsim_verbose:=false \
      > "${case_dir}/launch.log" 2>&1 &
  LAUNCH_PGID="$(pgid_of $!)"
  echo "  launch pgid=$LAUNCH_PGID -> ${case_dir}/launch.log"

  setsid ros2 bag record --use-sim-time -s mcap \
      -o "$bag_dir" "$INPUT_TOPIC" "$OUTPUT_TOPIC" \
      > "${case_dir}/bag.log" 2>&1 &
  BAG_PGID="$(pgid_of $!)"
  echo "  bag pgid=$BAG_PGID -> $bag_dir"

  # Fail fast: if the predictor node dies at startup, Gazebo keeps running and we
  # would otherwise record a full-length bag containing no /wave_predictions.
  local settle=0
  while (( settle < 40 )); do
    sleep 5
    settle=$((settle + 5))
    if grep -q "process has died.*the_next_wave_node" "${case_dir}/launch.log" 2>/dev/null; then
      echo "  ERROR: the_next_wave_node died at startup; see ${case_dir}/launch.log" >&2
      grep -A2 "process has died.*the_next_wave_node" "${case_dir}/launch.log" | head -5 >&2
      stop_pgid "$BAG_PGID" "bag recorder" 5 60
      BAG_PGID=""
      stop_pgid "$LAUNCH_PGID" "launch" 20 60
      LAUNCH_PGID=""
      return 1
    fi
    if ros2 topic info "$OUTPUT_TOPIC" 2>/dev/null | grep -q 'Publisher count: [1-9]'; then
      echo "  predictor node up, publishing on ${OUTPUT_TOPIC}"
      break
    fi
  done

  python3 "${SCRIPT_DIR}/wait_sim_time.py" --duration "$DURATION_SEC" 2>&1 | sed 's/^/  /'
  local wait_status="${PIPESTATUS[0]}"

  echo "  stopping recorder and sim..."
  stop_pgid "$BAG_PGID" "bag recorder" 5 60
  BAG_PGID=""
  stop_pgid "$LAUNCH_PGID" "launch" 20 60
  LAUNCH_PGID=""
  sleep 5

  if (( wait_status != 0 )); then
    echo "  WARNING: sim-time wait exited with status ${wait_status}; exporting whatever was recorded" >&2
  fi

  local export_args=(--prefix "${OUT_DIR}/${case_name}" --config "$config"
                     --input-topic "$INPUT_TOPIC" --output-topic "$OUTPUT_TOPIC")
  if [[ "$EXTRA_CSV" == true ]]; then
    export_args+=(--dense --spectrum)
  fi
  python3 "${SCRIPT_DIR}/bag_to_csv.py" "$bag_dir" "${export_args[@]}" 2>&1 | sed 's/^/  /'
  local export_status="${PIPESTATUS[0]}"

  if (( wait_status != 0 || export_status != 0 )); then
    return 1
  fi
  return 0
}

overall=0
IFS=',' read -ra CASE_LIST <<< "$CASES"
for case_name in "${CASE_LIST[@]}"; do
  run_case "$case_name" || overall=1
done

trap - INT TERM

echo
echo "=============================================================="
echo " CSVs in $OUT_DIR:"
ls -1 "$OUT_DIR"/*.csv 2>/dev/null | sed 's/^/   /' || echo "   (none)"
echo "=============================================================="
exit "$overall"
