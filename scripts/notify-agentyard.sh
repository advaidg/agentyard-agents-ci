#!/usr/bin/env bash
# notify-agentyard.sh — POST an image-push event to your AgentYard webhook.
#
# Use this locally (or from CI) after pushing an agent image to a Docker
# registry. AgentYard will link the image to the named agent, or hold the
# event until the agent registers (first boot). See
# /api/webhooks/agent-image for the receiver.
#
# Env:
#   AGENTYARD_BASE_URL     Base URL of your AgentYard gateway
#                          (e.g. http://localhost:8080)
#   YARD_WEBHOOK_SECRET    Shared secret configured on the registry
#
# Usage:
#   ./scripts/notify-agentyard.sh \
#       --agent triage-classifier \
#       --image docker.io/advaidg/agentyard2 \
#       --tag   triage-classifier-9f8a3c1 \
#       [--version 1.2.0] [--commit 9f8a3c1] [--pushed-by advaidg]

set -euo pipefail

agent=""; image=""; tag=""; version=""; commit=""; pushed_by="${USER:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --agent)     agent="$2";     shift 2 ;;
    --image)     image="$2";     shift 2 ;;
    --tag)       tag="$2";       shift 2 ;;
    --version)   version="$2";   shift 2 ;;
    --commit)    commit="$2";    shift 2 ;;
    --pushed-by) pushed_by="$2"; shift 2 ;;
    -h|--help)
      grep -E '^#' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

: "${AGENTYARD_BASE_URL:?set AGENTYARD_BASE_URL (e.g. http://localhost:8080)}"
: "${YARD_WEBHOOK_SECRET:?set YARD_WEBHOOK_SECRET (matches registry env)}"
: "${agent:?--agent is required}"
: "${image:?--image is required}"
: "${tag:?--tag is required}"

payload=$(jq -n \
  --arg agent_name "$agent" \
  --arg image      "$image" \
  --arg tag        "$tag" \
  --arg version    "$version" \
  --arg commit_sha "$commit" \
  --arg pushed_by  "$pushed_by" \
  '{agent_name:$agent_name,image:$image,tag:$tag,version:$version,commit_sha:$commit_sha,pushed_by:$pushed_by}')

curl -sS -X POST "$AGENTYARD_BASE_URL/api/webhooks/agent-image" \
  -H "X-AgentYard-Webhook-Secret: $YARD_WEBHOOK_SECRET" \
  -H 'Content-Type: application/json' \
  -d "$payload"
echo
