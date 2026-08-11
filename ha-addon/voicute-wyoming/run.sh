#!/bin/bash
# HA Add-on entrypoint — reads options from config and starts Wyoming service

MODEL_INFO="$(bashio::config 'model_info')"
MEL="$(bashio::config 'mel')"
THRESHOLD="$(bashio::config 'threshold')"
COOLDOWN="$(bashio::config 'cooldown')"
L1="$(bashio::config 'L1')"
L3="$(bashio::config 'L3')"
L5="$(bashio::config 'L5')"
DEBUG="$(bashio::config 'debug')"

ARGS=(
  --uri tcp://0.0.0.0:10400
  --model-info "$MODEL_INFO"
  --mel "$MEL"
  --threshold "$THRESHOLD"
  --cooldown "$COOLDOWN"
)

[[ "$L1" == "true" ]] && ARGS+=(--L1 1) || ARGS+=(--L1 0)
[[ "$L3" == "true" ]] && ARGS+=(--L3 1) || ARGS+=(--L3 0)
[[ "$L5" == "true" ]] && ARGS+=(--L5 1) || ARGS+=(--L5 0)
[[ "$DEBUG" == "true" ]] && ARGS+=(--debug)

bashio::log.info "Starting Voicute Wyoming service..."
bashio::log.info "Model: $MODEL_INFO  Threshold: $THRESHOLD  L1=$L1 L3=$L3 L5=$L5"

exec python wyoming/wyoming_voicute.py "${ARGS[@]}"
