#!/usr/bin/env bash
set -euo pipefail

BIND_HOST="127.0.0.1"
PORT="8050"
DEBUG="0"
NO_BROWSER="0"
WAIT_MODE="0"
TIMEOUT_SEC="30"

usage() {
  cat <<'EOF'
Usage: scripts/start_gui.sh [options]

Options:
  --host <host>        Bind host (default: 127.0.0.1)
  --port <port>        Bind port (default: 8050)
  --debug              Enable debug mode
  --no-browser         Do not open browser automatically
  --wait               Wait and keep foreground attached
  --timeout-sec <sec>  Startup timeout seconds (default: 30)
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      BIND_HOST="${2:-}"; shift 2 ;;
    --port)
      PORT="${2:-}"; shift 2 ;;
    --debug)
      DEBUG="1"; shift ;;
    --no-browser)
      NO_BROWSER="1"; shift ;;
    --wait)
      WAIT_MODE="1"; shift ;;
    --timeout-sec)
      TIMEOUT_SEC="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2 ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "uv command not found. Install uv and ensure it is on PATH." >&2
  exit 127
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv_cache}"
mkdir -p "$UV_CACHE_DIR"

LIBGOMP_DIR="$ROOT_DIR/.vendor/libgomp/usr/lib/x86_64-linux-gnu"
if [[ -d "$LIBGOMP_DIR" ]]; then
  export LD_LIBRARY_PATH="$LIBGOMP_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

test_tcp_port() {
  local host="$1"
  local port="$2"
  if timeout 0.25 bash -c "</dev/tcp/${host}/${port}" >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

UV_ARGS=(run veldra-gui --host "$BIND_HOST" --port "$PORT")
if [[ "$DEBUG" == "1" ]]; then
  UV_ARGS+=(--debug)
fi

echo "Starting Veldra GUI: uv ${UV_ARGS[*]}"
uv "${UV_ARGS[@]}" &
SERVER_PID=$!

URL="http://${BIND_HOST}:${PORT}"
DEADLINE=$((SECONDS + TIMEOUT_SEC))

while (( SECONDS < DEADLINE )); do
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    wait "$SERVER_PID" || true
    echo "GUI server exited early." >&2
    exit 1
  fi
  if test_tcp_port "$BIND_HOST" "$PORT"; then
    if [[ "$NO_BROWSER" != "1" ]] && command -v xdg-open >/dev/null 2>&1; then
      xdg-open "$URL" >/dev/null 2>&1 || true
    fi
    echo "GUI is ready: $URL"
    break
  fi
  sleep 0.3
done

if ! test_tcp_port "$BIND_HOST" "$PORT"; then
  if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  echo "GUI did not become ready within ${TIMEOUT_SEC}s." >&2
  exit 1
fi

if [[ "$WAIT_MODE" == "1" ]]; then
  echo "Press Ctrl+C to stop the GUI server."
  trap 'kill "$SERVER_PID" >/dev/null 2>&1 || true' INT TERM EXIT
  wait "$SERVER_PID"
else
  echo "Server process id: $SERVER_PID"
fi
