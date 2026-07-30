# agentyard-agents-ci

The 10 AgentYard demo agents (triage pipeline + enterprise document agents),
each independently built and published by a CI pipeline the platform
provides — not a one-off script. Push a change to `agents/<name>/`, and
that agent alone gets linted, built, smoke-tested, pushed to ECR, and
announced back to your AgentYard registry.

## Layout

```
agents/
  triage-classifier/
    Dockerfile
    main.py
  triage-sentiment/
  ...
vendor-agentyard-sdk/   # vendored copy of AgentYard's v2 SDK
scripts/
  notify-agentyard.sh   # manual replay of the webhook-announce step
```

## What the pipeline does (`.github/workflows/ci.yml`)

Only agents whose files actually changed get rebuilt — a shared-file
change (the vendored SDK, the workflow itself) rebuilds everything.

1. **Lint** — `python -m py_compile` on the changed agent's `main.py`.
   Fails fast before spending a build.
2. **Build** — multi-arch (`linux/amd64` + `linux/arm64`) Docker image via
   Buildx, layer-cached per agent.
3. **Smoke test** — runs the freshly-built image, waits for `/health` to
   respond, fails the job if the container doesn't come up clean. Catches
   a broken image before it's ever pushed anywhere.
4. **Push** — to ECR, `main` branch only. Pull requests build and
   smoke-test but never push, so a PR can't accidentally publish.
5. **Announce** — `POST /api/webhooks/agent-image` on your AgentYard
   instance, so the registry's `docker_image` field updates automatically
   the moment CI lands. No manual sync step.

## One-time setup

**AWS side** (already done for this repo — see AgentYard's own
`infra/terraform/eks/` session notes): a GitHub OIDC role scoped to
`repo:advaidg/agentyard-agents-ci:*` with ECR push rights, no static AWS
keys stored anywhere.

**Repo secrets/vars** (Settings → Secrets and variables → Actions):

| Name | Type | Value |
|---|---|---|
| `AWS_ROLE_ARN` | variable | IAM role ARN for OIDC push |
| `AWS_REGION` | variable | `us-east-1` |
| `ECR_REGISTRY` | variable | `<account-id>.dkr.ecr.us-east-1.amazonaws.com` |
| `AGENTYARD_WEBHOOK_URL` | variable | `https://<your-agentyard>/api/webhooks/agent-image` |
| `YARD_WEBHOOK_SECRET` | secret | matches `YARD_WEBHOOK_SECRET` on the registry service |

Leave `AGENTYARD_WEBHOOK_URL` unset and the workflow prints a
copy-pasteable `curl` command instead of failing — useful for a fully
local AgentYard instance CI can't reach.

## Manually replaying an announce

```bash
AGENTYARD_BASE_URL=http://localhost:8080 \
YARD_WEBHOOK_SECRET=... \
./scripts/notify-agentyard.sh \
  --agent triage-classifier \
  --image 123456789.dkr.ecr.us-east-1.amazonaws.com/agentyard-triage-classifier \
  --tag latest
```
