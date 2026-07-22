# OpenRouter staging model experiment

The OpenRouter models and named A/B/C profiles are disabled unless both
`ENABLE_OPENROUTER_MODEL_EXPERIMENT=true` and `OPENROUTER_API_KEY` are present.
Production must keep the flag disabled until the benchmark is reviewed and a
separate production rollout is explicitly approved.

Every OpenRouter request requires strict structured output, zero-data-retention,
and denied provider data collection. The benchmark additionally pins the
provider selected by its conformance preflight and disables provider fallback.

## Staging configuration

Set the following on both `startup-ranker-web` and `startup-ranker-worker` in
the Railway `staging` environment:

```text
APP_ENV=staging
ENABLE_OPENROUTER_MODEL_EXPERIMENT=true
OPENROUTER_API_KEY=<managed secret>
```

Do not change the default model profile during deployment. Confirm
`/api/config` exposes `gpt_current`, `kimi_k26`, and `glm_deepseek_flash`, then
smoke-test one public company through each profile.

## Frozen campaign

Run these commands with staging environment variables and `PYTHONPATH=src`.
Artifacts stay under the ignored `exports/` directory and are never inserted
into Deal Intelligence application tables.

```bash
python -m agent.model_benchmark prepare --campaign-dir exports/model-benchmark/<campaign>
python -m agent.model_benchmark approve --campaign-dir exports/model-benchmark/<campaign> --approved-by Dusan
python -m agent.model_benchmark preflight --campaign-dir exports/model-benchmark/<campaign>
python -m agent.model_benchmark run --campaign-dir exports/model-benchmark/<campaign>
```

The prepare step reads the ten companies from job `f20aa510`, plus the newest
distinct completed pitch-deck and Specter companies. It freezes company data,
evidence chunks, available source files, hashes, the fixed randomized schedule,
and an approval-pending manifest. Approval verifies all hashes before allowing
any evidence to be sent to OpenRouter.

The run step is resumable and sequential. It creates 72 internal run reports,
summary JSON/CSV, a protected blinding key, output-only review files, and two
reviewer score sheets. Web search and live Specter MCP are disabled.

Reviewers score factual support, material completeness, investment insight,
pro/con balance, ranking calibration, company rank, unsupported claims, and
systemic omissions. Disagreements greater than one point or conflicting
unsupported-claim flags require an adjudication CSV before unblinding:

```bash
python -m agent.model_benchmark evaluate \
  --campaign-dir exports/model-benchmark/<campaign> \
  --adjudication exports/model-benchmark/<campaign>/adjudication.csv
```

`decision.json` applies the pre-declared quality, reliability, cost, and speed
gates. It never changes a staging or production default. Passing candidates
must next complete five fresh-company staging shadow runs with normal web and
Specter integrations before any separately approved production canary.
