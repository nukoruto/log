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
referer: referer
user_agent: user_agent
ip: ip
op_category: category
MAP

run_step() {
  local subcommand="$1"
  shift
  "${PYTHON_BIN}" -m ds_contract.cli --seed "${SEED}" "${subcommand}" "$@"
}

run_step validate "${RAW_CSV}" --map "${MAP_YAML}" --out "${CONTRACT_CSV}" --meta "${META_VALIDATE}"
run_step sessionize "${CONTRACT_CSV}" --out "${SESSIONED_CSV}" --meta "${META_SESSION}"
run_step deltify "${SESSIONED_CSV}" --out "${DELTIFIED_CSV}" --meta "${META_DT}"

echo "quickstart outputs ready at ${WORKDIR}"
