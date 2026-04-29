#!/usr/bin/env bash
set -euo pipefail

TP2_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP2_OPS_DIR="$(cd "${TP2_COMMON_DIR}/.." && pwd)"
TP2_REPO_DIR="$(cd "${TP2_OPS_DIR}/.." && pwd)"

tp2_load_config() {
  local explicit_config="${TP2_LAB_CONFIG:-}"
  local candidates=()

  if [[ -n "${explicit_config}" ]]; then
    candidates+=("${explicit_config}")
  fi

  candidates+=(
    "/etc/tp2/lab.env"
    "${HOME}/.config/tp2/lab.env"
    "${TP2_OPS_DIR}/tp2-lab.env"
    "${TP2_OPS_DIR}/tp2-lab.env.example"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      # shellcheck disable=SC1090
      source "${candidate}"
      TP2_CONFIG_FILE="${candidate}"
      break
    fi
  done

  : "${TP2_EPC_SSH:=tp2@100.97.19.112}"
  : "${TP2_DEFAULT_PROFILE:=jetson}"
  : "${TP2_ENB_SSH:=tp2@10.10.10.2}"
  : "${TP2_ENB_SSH_FALLBACK:=}"
  : "${TP2_ENB_SSH_PROXY:=tp2@100.97.19.112}"
  : "${TP2_JETSON_SSH:=grupo4@100.115.99.8}"

  : "${TP2_EPC_BACKHAUL_IP:=10.10.10.1}"
  : "${TP2_ENB_BACKHAUL_IP:=10.10.10.2}"
  : "${TP2_EPC_SGI_IP:=172.16.0.1}"
  : "${TP2_CAR_UE_IP:=172.16.0.2}"
  : "${TP2_CAR_SSH_USER:=}"
  : "${TP2_CAR_SYSTEMD_SERVICE:=ARTEMIS.service}"
  : "${TP2_JETSON_INFERENCE_URL:=http://100.115.99.8:9001}"
  : "${TP2_ROBOFLOW_MODEL_ID:=tp2-g4-2026/2}"

  : "${TP2_EPC_SRSEPC_SERVICE:=tp2-srsepc.service}"
  : "${TP2_ENB_LINK_SERVICE:=tp2-enb-link.service}"
  : "${TP2_ENB_FPGA_SERVICE:=tp2-bladerf-fpga.service}"
  : "${TP2_ENB_SRSENB_SERVICE:=tp2-srsenb.service}"
  : "${TP2_EPC_SRSEPC_CMD:=/usr/local/bin/srsepc /home/tp2/.config/srsran/epc.conf}"
  : "${TP2_ENB_SRSENB_CMD:=/usr/local/bin/srsenb /home/tp2/.config/srsran/enb.conf}"
  : "${TP2_EPC_MQTT_SERVICE:=mosquitto}"
  : "${TP2_EPC_LOCAL_INFERENCE_SERVICE:=tp2-local-inference.service}"
  : "${TP2_EPC_CAR_CONTROL_SERVICE:=tp2-car-control.service}"
  : "${TP2_EPC_AM_CLOUD_SERVICE:=tp2-car-command-am-cloud.service}"
  : "${TP2_JETSON_INFERENCE_SERVICE:=tp2-roboflow-inference.service}"

  : "${TP2_START_LOCAL_INFERENCE_FALLBACK:=1}"
  : "${TP2_START_JETSON_INFERENCE:=1}"
  : "${TP2_STOP_MOSQUITTO_ON_DOWN:=0}"
  : "${TP2_STOP_JETSON_ON_DOWN:=0}"
  : "${TP2_REQUIRE_CAR_UE:=0}"
  : "${TP2_RESTART_CAR_ON_UP:=0}"
  : "${TP2_CAR_WEB_URL:=http://127.0.0.1:8088/status.json}"
  : "${TP2_CAR_WEB_PUBLIC_URL:=http://100.97.19.112:8088/}"
  : "${TP2_CHECK_CAR_WEB:=1}"

  : "${TP2_PUBLISH_CAR_MODE_ON_UP:=1}"
  : "${TP2_MQTT_CLEAR_RETAINED_ON_UP:=1}"
  : "${TP2_MQTT_RETAIN_COMMAND:=0}"
  : "${TP2_MQTT_HOST:=172.16.0.1}"
  : "${TP2_MQTT_PORT:=1883}"
  : "${TP2_MQTT_COMMAND_TOPIC:=1/command}"
  : "${TP2_MQTT_COMMAND_PAYLOAD:=AM-Cloud}"
  : "${TP2_MQTT_SUB_TIMEOUT_SEC:=2}"

  : "${TP2_WAIT_SSH_TIMEOUT_SEC:=20}"
  : "${TP2_SSH_CONNECT_TIMEOUT_SEC:=10}"
  : "${TP2_WAIT_SERVICE_TIMEOUT_SEC:=45}"
  : "${TP2_WAIT_LTE_TIMEOUT_SEC:=90}"
  : "${TP2_WAIT_UE_TIMEOUT_SEC:=120}"
  : "${TP2_WAIT_HTTP_TIMEOUT_SEC:=60}"
  : "${TP2_CAR_ATTACH_DELAY_SEC:=30}"
  : "${TP2_CAR_RESTART_DELAY_SEC:=10}"
  : "${TP2_CAR_SSH_SCAN_START:=2}"
  : "${TP2_CAR_SSH_SCAN_END:=20}"
  : "${TP2_CAR_SSH_DISCOVERY_TIMEOUT_SEC:=2}"
  : "${TP2_CAR_SSH_PASSWORD:=}"
}

tp2_log() {
  printf '[tp2] %s\n' "$*"
}

tp2_warn() {
  printf '[tp2][warn] %s\n' "$*" >&2
}

tp2_die() {
  printf '[tp2][error] %s\n' "$*" >&2
  exit 1
}

tp2_is_local_target() {
  [[ "$1" == "local" || "$1" == "localhost" || "$1" == "127.0.0.1" ]]
}

tp2_ssh() {
  local target="$1"
  shift

  local ssh_args=(
    -o BatchMode=yes
    -o ConnectTimeout="${TP2_SSH_CONNECT_TIMEOUT_SEC}"
    -o ServerAliveInterval=5
    -o ServerAliveCountMax=2
    -o StrictHostKeyChecking=accept-new
  )

  ssh "${ssh_args[@]}" "${target}" "$@"
}

tp2_scp_to() {
  local source="$1"
  local target="$2"
  local dest="$3"

  local scp_args=(
    -o BatchMode=yes
    -o ConnectTimeout="${TP2_SSH_CONNECT_TIMEOUT_SEC}"
    -o StrictHostKeyChecking=accept-new
  )

  if tp2_is_local_target "${target}"; then
    cp "${source}" "${dest}"
    return $?
  fi

  if [[ "${target}" == "${TP2_ENB_SSH}" && -n "${TP2_ENB_SSH_PROXY:-}" ]]; then
    local stage="/tmp/tp2-stage-$(basename "${dest}").$$"
    local stage_q
    local dest_q
    printf -v stage_q "%q" "${stage}"
    printf -v dest_q "%q" "${dest}"

    scp "${scp_args[@]}" "${source}" "${TP2_ENB_SSH_PROXY}:${stage}"
    tp2_ssh "${TP2_ENB_SSH_PROXY}" \
      "scp -o BatchMode=yes -o ConnectTimeout=${TP2_SSH_CONNECT_TIMEOUT_SEC} -o StrictHostKeyChecking=accept-new ${stage_q} ${target}:${dest_q} && rm -f ${stage_q}"
    return $?
  fi

  if scp "${scp_args[@]}" "${source}" "${target}:${dest}"; then
    return 0
  fi

  if [[
    "${target}" == "${TP2_ENB_SSH}" &&
    -n "${TP2_ENB_SSH_FALLBACK:-}" &&
    "${TP2_ENB_SSH_FALLBACK}" != "${TP2_ENB_SSH}"
  ]]; then
    tp2_warn "SCP to ${TP2_ENB_SSH} failed; retrying eNodeB fallback ${TP2_ENB_SSH_FALLBACK}"
    scp "${scp_args[@]}" "${source}" "${TP2_ENB_SSH_FALLBACK}:${dest}"
    return $?
  fi

  return 1
}

tp2_remote_sh() {
  local target="$1"
  shift
  local command="$*"

  if tp2_is_local_target "${target}"; then
    bash -lc "${command}"
    return $?
  fi

  if [[ "${target}" == "${TP2_ENB_SSH}" && -n "${TP2_ENB_SSH_PROXY:-}" ]]; then
    tp2_ssh "${TP2_ENB_SSH_PROXY}" \
      "ssh -o BatchMode=yes -o ConnectTimeout=${TP2_SSH_CONNECT_TIMEOUT_SEC} -o StrictHostKeyChecking=accept-new ${target} 'bash -s'" \
      <<<"${command}"
    return $?
  fi

  tp2_ssh "${target}" "bash -s" <<<"${command}"
}

tp2_remote_systemctl() {
  local target="$1"
  local action="$2"
  local unit="$3"

  if [[ "${unit}" == "mosquitto" && ( "${action}" == "start" || "${action}" == "stop" ) ]]; then
    tp2_remote_sh "${target}" "sudo -n systemctl ${action} mosquitto || sudo -n systemctl ${action} mosquitto.service"
    return
  fi

  tp2_remote_sh "${target}" "sudo -n systemctl ${action} ${unit}"
}

tp2_remote_systemctl_quiet() {
  local target="$1"
  local action="$2"
  local unit="$3"

  if [[ "${unit}" == "mosquitto" && ( "${action}" == "start" || "${action}" == "stop" ) ]]; then
    tp2_remote_sh "${target}" "sudo -n systemctl ${action} mosquitto >/dev/null 2>&1 || sudo -n systemctl ${action} mosquitto.service >/dev/null 2>&1 || true"
    return
  fi

  tp2_remote_sh "${target}" "sudo -n systemctl ${action} ${unit} >/dev/null 2>&1 || true"
}

tp2_remote_systemctl_is_active() {
  local target="$1"
  local unit="$2"

  tp2_remote_sh "${target}" "systemctl is-active --quiet ${unit}"
}

tp2_remote_can_sudo_systemctl() {
  local target="$1"
  local action="$2"
  local unit="$3"

  tp2_remote_sh "${target}" "sudo -n -l /usr/bin/systemctl ${action} ${unit} >/dev/null"
}

tp2_mqtt_retained_command_payload() {
  local host_q
  local port_q
  local topic_q
  local timeout_q

  printf -v host_q "%q" "${TP2_MQTT_HOST}"
  printf -v port_q "%q" "${TP2_MQTT_PORT}"
  printf -v topic_q "%q" "${TP2_MQTT_COMMAND_TOPIC}"
  printf -v timeout_q "%q" "${TP2_MQTT_SUB_TIMEOUT_SEC}"

  tp2_remote_sh "${TP2_EPC_SSH}" "
timeout ${timeout_q} mosquitto_sub -h ${host_q} -p ${port_q} -t ${topic_q} -C 1 2>/dev/null || true
"
}

tp2_clear_retained_car_mode() {
  local host_q
  local port_q
  local topic_q

  printf -v host_q "%q" "${TP2_MQTT_HOST}"
  printf -v port_q "%q" "${TP2_MQTT_PORT}"
  printf -v topic_q "%q" "${TP2_MQTT_COMMAND_TOPIC}"

  tp2_remote_sh "${TP2_EPC_SSH}" \
    "mosquitto_pub -q 1 -r -n -h ${host_q} -p ${port_q} -t ${topic_q}"
}

tp2_prepare_car_mode_topic() {
  [[ "${TP2_MQTT_CLEAR_RETAINED_ON_UP}" == "1" ]] || return 0

  local retained
  retained="$(tp2_mqtt_retained_command_payload | head -n 1 || true)"
  if [[ -z "${retained}" ]]; then
    tp2_log "MQTT ${TP2_MQTT_COMMAND_TOPIC}: no retained command to clear"
    return 0
  fi

  tp2_log "MQTT ${TP2_MQTT_COMMAND_TOPIC}: clearing retained command before startup publish"
  tp2_clear_retained_car_mode
}

tp2_publish_car_mode_once() {
  [[ "${TP2_PUBLISH_CAR_MODE_ON_UP}" == "1" ]] || {
    tp2_log "Skipping car mode publish because TP2_PUBLISH_CAR_MODE_ON_UP=0"
    return 0
  }

  local host_q
  local port_q
  local topic_q
  local payload_q
  local retain_arg=""

  printf -v host_q "%q" "${TP2_MQTT_HOST}"
  printf -v port_q "%q" "${TP2_MQTT_PORT}"
  printf -v topic_q "%q" "${TP2_MQTT_COMMAND_TOPIC}"
  printf -v payload_q "%q" "${TP2_MQTT_COMMAND_PAYLOAD}"
  if [[ "${TP2_MQTT_RETAIN_COMMAND}" == "1" ]]; then
    retain_arg="-r "
    tp2_warn "Publishing retained car mode because TP2_MQTT_RETAIN_COMMAND=1"
  fi

  tp2_remote_sh "${TP2_EPC_SSH}" \
    "mosquitto_pub -q 1 ${retain_arg}-h ${host_q} -p ${port_q} -t ${topic_q} -m ${payload_q}"
}

tp2_remote_has_process_cmd() {
  local target="$1"
  local expected_cmd="$2"
  local expected_q

  printf -v expected_q "%q" "${expected_cmd}"
  tp2_remote_sh "${target}" \
    "ps -eo args= | grep -F -- ${expected_q} | grep -v -F 'grep -F --' >/dev/null"
}

tp2_wait_process_cmd() {
  local target="$1"
  local timeout_sec="$2"
  local message="$3"
  local expected_cmd="$4"

  tp2_wait_until "${timeout_sec}" "${message}" \
    tp2_remote_has_process_cmd "${target}" "${expected_cmd}"
}

tp2_wait_until() {
  local timeout_sec="$1"
  local message="$2"
  shift 2

  local deadline=$((SECONDS + timeout_sec))
  while (( SECONDS < deadline )); do
    if "$@" >/dev/null 2>&1; then
      tp2_log "${message}: ok"
      return 0
    fi
    sleep 2
  done

  tp2_die "${message}: timeout after ${timeout_sec}s"
}

tp2_wait_remote() {
  local target="$1"
  local timeout_sec="$2"
  local message="$3"
  local command="$4"

  tp2_wait_until "${timeout_sec}" "${message}" tp2_remote_sh "${target}" "${command}"
}

tp2_wait_remote_optional() {
  local target="$1"
  local timeout_sec="$2"
  local message="$3"
  local command="$4"
  local deadline=$((SECONDS + timeout_sec))

  while (( SECONDS < deadline )); do
    if tp2_remote_sh "${target}" "${command}" >/dev/null 2>&1; then
      tp2_log "${message}: ok"
      return 0
    fi
    sleep 2
  done

  return 1
}

tp2_wait_service_active() {
  local target="$1"
  local unit="$2"
  local timeout_sec="${3:-${TP2_WAIT_SERVICE_TIMEOUT_SEC}}"

  tp2_wait_remote "${target}" "${timeout_sec}" "${target} ${unit} active" \
    "systemctl is-active --quiet ${unit}"
}

tp2_check_bladerf_cli_released_once() {
  tp2_remote_sh "${TP2_ENB_SSH}" \
    "! pgrep -x bladeRF-cli >/dev/null && ! pgrep -x bladerf-cli >/dev/null" \
    >/dev/null 2>&1
}

tp2_wait_bladerf_cli_released() {
  local timeout_sec="${1:-${TP2_WAIT_SERVICE_TIMEOUT_SEC}}"

  tp2_wait_until "${timeout_sec}" "bladeRF-cli released the SDR" \
    tp2_check_bladerf_cli_released_once
}

tp2_check_s1_once() {
  tp2_remote_sh "${TP2_EPC_SSH}" \
    "ss -H -nA sctp | grep ESTAB | grep -F ':36412' | grep -F '${TP2_ENB_BACKHAUL_IP}:' >/dev/null" \
    >/dev/null 2>&1
}

tp2_wait_s1() {
  local timeout_sec="${1:-${TP2_WAIT_LTE_TIMEOUT_SEC}}"

  tp2_wait_until "${timeout_sec}" "EPC <-> eNodeB S1 association" \
    tp2_check_s1_once
}

tp2_require_expect() {
  command -v expect >/dev/null 2>&1 || tp2_die "expect is required for car SSH automation"
}

tp2_find_car_ssh_ip_once() {
  local ue_ip_q
  local sgi_prefix_q
  local scan_start_q
  local scan_end_q
  local scan_timeout_q

  printf -v ue_ip_q "%q" "${TP2_CAR_UE_IP}"
  printf -v sgi_prefix_q "%q" "${TP2_EPC_SGI_IP%.*}"
  printf -v scan_start_q "%q" "${TP2_CAR_SSH_SCAN_START}"
  printf -v scan_end_q "%q" "${TP2_CAR_SSH_SCAN_END}"
  printf -v scan_timeout_q "%q" "${TP2_CAR_SSH_DISCOVERY_TIMEOUT_SEC}"

  tp2_remote_sh "${TP2_EPC_SSH}" "
command -v ssh-keyscan >/dev/null 2>&1 || exit 127
ue_ip=${ue_ip_q}
sgi_prefix=${sgi_prefix_q}
scan_start=${scan_start_q}
scan_end=${scan_end_q}
scan_timeout=${scan_timeout_q}

if ssh-keyscan -T \"\${scan_timeout}\" \"\${ue_ip}\" >/dev/null 2>&1; then
  printf '%s\n' \"\${ue_ip}\"
  exit 0
fi

for host in \$(seq \"\${scan_start}\" \"\${scan_end}\"); do
  candidate=\"\${sgi_prefix}.\${host}\"
  if [[ \"\${candidate}\" == \"\${ue_ip}\" ]]; then
    continue
  fi
  if ssh-keyscan -T \"\${scan_timeout}\" \"\${candidate}\" >/dev/null 2>&1; then
    printf '%s\n' \"\${candidate}\"
    exit 0
  fi
done

exit 1
"
}

tp2_wait_car_ssh_ip() {
  local timeout_sec="${1:-${TP2_WAIT_UE_TIMEOUT_SEC}}"
  local deadline=$((SECONDS + timeout_sec))
  local car_ip=""

  while (( SECONDS < deadline )); do
    if car_ip="$(tp2_find_car_ssh_ip_once 2>/dev/null)" && [[ -n "${car_ip}" ]]; then
      printf '%s\n' "${car_ip}"
      return 0
    fi
    sleep 2
  done

  return 1
}

tp2_restart_car_service_via_epc() {
  local car_ip="$1"

  [[ -n "${TP2_CAR_SSH_USER:-}" ]] || tp2_die "TP2_CAR_SSH_USER is required to restart ${TP2_CAR_SYSTEMD_SERVICE} on the car"
  [[ -n "${TP2_CAR_SSH_PASSWORD:-}" ]] || tp2_die "TP2_CAR_SSH_PASSWORD is required to restart ${TP2_CAR_SYSTEMD_SERVICE} on the car"
  tp2_require_expect

  tp2_log "Restarting ${TP2_CAR_SYSTEMD_SERVICE} on ${TP2_CAR_SSH_USER}@${car_ip} via EPC"
  if ! EXPECT_EPC_SSH="${TP2_EPC_SSH}" \
       EXPECT_CAR_SSH_USER="${TP2_CAR_SSH_USER}" \
       EXPECT_CAR_SSH_IP="${car_ip}" \
       EXPECT_CAR_SERVICE="${TP2_CAR_SYSTEMD_SERVICE}" \
       EXPECT_CAR_PASSWORD="${TP2_CAR_SSH_PASSWORD}" \
       EXPECT_SSH_TIMEOUT="${TP2_SSH_CONNECT_TIMEOUT_SEC}" \
       expect <<'EOF'
set timeout 60
set epc $env(EXPECT_EPC_SSH)
set car_user $env(EXPECT_CAR_SSH_USER)
set car_ip $env(EXPECT_CAR_SSH_IP)
set car_service $env(EXPECT_CAR_SERVICE)
set car_password $env(EXPECT_CAR_PASSWORD)
set ssh_timeout $env(EXPECT_SSH_TIMEOUT)

set nested_cmd [format {ssh -tt -o StrictHostKeyChecking=accept-new -o PreferredAuthentications=password -o PubkeyAuthentication=no -o ConnectTimeout=%s %s@%s 'sudo systemctl restart %s && sudo systemctl is-active --quiet %s'} $ssh_timeout $car_user $car_ip $car_service $car_service]
spawn ssh -tt -o StrictHostKeyChecking=accept-new -o ConnectTimeout=$ssh_timeout $epc $nested_cmd
expect {
  -re {(?i)yes/no} { send "yes\r"; exp_continue }
  -re {(?i)password.*:} { send "$car_password\r"; exp_continue }
  eof
}
catch wait result
if {[llength $result] >= 4} {
  exit [lindex $result 3]
}
exit 0
EOF
  then
    tp2_die "Could not restart ${TP2_CAR_SYSTEMD_SERVICE} on ${TP2_CAR_SSH_USER}@${car_ip} via EPC; check TP2_CAR_SSH_USER, TP2_CAR_SSH_PASSWORD, and car sudo access"
  fi

  tp2_log "Waiting ${TP2_CAR_RESTART_DELAY_SEC}s after car ${TP2_CAR_SYSTEMD_SERVICE} restart"
  sleep "${TP2_CAR_RESTART_DELAY_SEC}"
}

tp2_wait_car_ue() {
  local timeout_sec="${1:-${TP2_WAIT_UE_TIMEOUT_SEC}}"

  tp2_wait_remote "${TP2_EPC_SSH}" "${timeout_sec}" "car UE ${TP2_CAR_UE_IP} reachable or recently attached" \
    "ping -c 1 -W 1 ${TP2_CAR_UE_IP} >/dev/null 2>&1 || grep -q 'UE IP: ${TP2_CAR_UE_IP}' /srv/tp2/logs/srsepc.log"
}

tp2_check_car_ue_once() {
  tp2_remote_sh "${TP2_EPC_SSH}" \
    "ping -c 1 -W 1 ${TP2_CAR_UE_IP} >/dev/null 2>&1 || grep -q 'UE IP: ${TP2_CAR_UE_IP}' /srv/tp2/logs/srsepc.log" \
    >/dev/null 2>&1
}

tp2_maybe_wait_car_ue() {
  local message="$1"

  if [[ "${TP2_REQUIRE_CAR_UE}" == "1" ]]; then
    tp2_log "${message}"
    tp2_wait_car_ue "${TP2_WAIT_UE_TIMEOUT_SEC}"
    return
  fi

  if tp2_check_car_ue_once; then
    tp2_log "${message}: ok"
  else
    tp2_warn "${message}: not confirmed; continuing because TP2_REQUIRE_CAR_UE=0"
  fi
}

tp2_require_ssh() {
  local target="$1"
  local timeout_sec="${2:-${TP2_WAIT_SSH_TIMEOUT_SEC}}"

  tp2_wait_until "${timeout_sec}" "ssh ${target}" tp2_remote_sh "${target}" "true"
}

tp2_http_host_port_from_url() {
  python3 - "$1" <<'PY'
import sys
from urllib.parse import urlparse

parsed = urlparse(sys.argv[1])
if not parsed.hostname or not parsed.port:
    raise SystemExit(1)
print(parsed.hostname)
print(parsed.port)
PY
}
