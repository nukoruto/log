#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
WORKDIR_INPUT="${1:-}" 
if [[ -n "${WORKDIR_INPUT}" ]]; then
  WORKDIR="${WORKDIR_INPUT}"
  case "${WORKDIR}" in
    /*) : ;;
    *) WORKDIR="${ROOT_DIR}/${WORKDIR}" ;;
  esac
else
  WORKDIR="${ROOT_DIR}/artifacts/quickstart"
fi

SEED="${QUICKSTART_SEED:-202401}"
PYTHON_BIN="${PYTHON:-python}"

if [[ -e "${WORKDIR}" ]]; then
  rm -rf "${WORKDIR}"
fi
mkdir -p "${WORKDIR}"

RAW_CSV="${WORKDIR}/raw.csv"
MAP_YAML="${WORKDIR}/map.yaml"
CONTRACT_CSV="${WORKDIR}/contract.csv"
SESSIONED_CSV="${WORKDIR}/sessioned.csv"
DELTIFIED_CSV="${WORKDIR}/deltified.csv"
META_VALIDATE="${WORKDIR}/meta_validate.json"
META_SESSION="${WORKDIR}/meta_session.json"
META_DT="${WORKDIR}/meta_dt.json"

cat <<'DATA' >"${RAW_CSV}"
timestamp,user,session,method,path,referer,user_agent,ip,category
2024-04-01T00:00:05+00:00,user-1,s-1,GET,/index,-,Mozilla/5.0,198.51.100.10,read
2024-04-01T00:00:55+00:00,user-2,s-5,GET,/dashboard,-,Mozilla/5.0,198.51.100.11,read
2024-04-01T00:01:25+00:00,user-1,s-1,POST,/submit,https://origin.example,Requests/2.31,198.51.100.10,write
2024-04-01T00:05:55+00:00,user-2,s-5,GET,/reports,-,Mozilla/5.0,198.51.100.11,read
2024-04-01T02:15:25+00:00,user-1,s-1,GET,/index,-,Mozilla/5.0,198.51.100.10,read
2024-04-01T12:05:55+00:00,user-2,s-5,GET,/logout,-,Mozilla/5.0,198.51.100.11,logout
DATA

cat <<'MAP' >"${MAP_YAML}"
timestamp_utc: timestamp
uid: user
session_id: session
method: method
path: path
user_agent: user_agent
ip: ip
op_category: category
MAP

run_contract() {
  local subcommand="$1"
  shift
  "${PYTHON_BIN}" -m ds_contract.cli --seed "${SEED}" "${subcommand}" "$@"
}

run_scenario() {
  local subcommand="$1"
  shift
  "${PYTHON_BIN}" -m scenario_design.cli --seed "${SEED}" "${subcommand}" "$@"
}

run_generator() {
  local subcommand="$1"
  shift
  "${PYTHON_BIN}" -m log_generator.cli --seed "${SEED}" "${subcommand}" "$@"
}

run_models() {
  local subcommand="$1"
  shift
  "${PYTHON_BIN}" -m models_lstm.cli "${subcommand}" --seed "${SEED}" "$@"
}

# 1. Contract
run_contract validate "${RAW_CSV}" --map "${MAP_YAML}" --out "${CONTRACT_CSV}" --meta "${META_VALIDATE}"
run_contract sessionize "${CONTRACT_CSV}" --out "${SESSIONED_CSV}" --meta "${META_SESSION}"
run_contract deltify "${SESSIONED_CSV}" --out "${DELTIFIED_CSV}" --meta "${META_DT}"

# 2. Scenario Design
STATS_PKL="${WORKDIR}/stats.pkl"
SPEC_JSON="${WORKDIR}/spec.json"

run_scenario fit "${DELTIFIED_CSV}" --out "${STATS_PKL}"
# Plan with a small time injection anomaly for testing
run_scenario plan --stats "${STATS_PKL}" --out "${SPEC_JSON}" --anom "time(mode=propagate,p=0.1)"

# 3. Log Generator
GEN_NORMAL_CSV="${WORKDIR}/gen_normal.csv"
GEN_ANOM_CSV="${WORKDIR}/gen_anom.csv"
GEN_AUDIT_JSONL="${WORKDIR}/gen_audit.jsonl"
GEN_META_JSON="${WORKDIR}/gen_meta.json"

# Override start time (t0) to match sample data range or arbitrary future
run_generator run --spec "${SPEC_JSON}" \
  --normal "${GEN_NORMAL_CSV}" \
  --anom "${GEN_ANOM_CSV}" \
  --audit "${GEN_AUDIT_JSONL}" \
  --meta "${GEN_META_JSON}" \
  --t0 "2024-04-01T00:00:00+00:00"

# 4. Models (LSTM)
MODEL_DIR="${WORKDIR}/models"
SCORED_CSV="${WORKDIR}/scored.csv"

# Train on generated normal data
# Using small parameters for quickstart speed
run_models train \
  --normal "${GEN_NORMAL_CSV}" \
  --val "${GEN_NORMAL_CSV}" \
  --out "${MODEL_DIR}" \
  --epochs 2 \
  --batch-size 4 \
  --hidden-dim 16

# Score the generated anomalous data
run_models score \
  --model "${MODEL_DIR}/best.ckpt" \
  --in "${GEN_ANOM_CSV}" \
  --out "${SCORED_CSV}"

echo "quickstart pipeline complete!"
echo "Outputs at ${WORKDIR}"
echo "  - Spec: ${SPEC_JSON}"
echo "  - Gen Logs: ${GEN_NORMAL_CSV}, ${GEN_ANOM_CSV}"
echo "  - Model: ${MODEL_DIR}/best.ckpt"
echo "  - Scored: ${SCORED_CSV}"
