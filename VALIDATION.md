# Validation record

On 2026-09-27, the credential-free DuckDB integration suite and Snowflake
runtime isolation tests passed all nine tests. `elt-rlvr demo-local` executed
extraction/loading, a transformation, and terminal grading with reward `1.0`.
The Snowflake tests use fake credentials and stubbed external services; they
verify that EL credentials affect only the private rollout copy and that
copied secrets are cleaned on startup or Docker removal failures.

One Tinker RL optimization step completed with the repository's local fixture:

```bash
python -m elt_rlvr.tinker_recipe --max-steps 1 --group-size 4
```

The run used `Qwen/Qwen3.5-4B`. All four model rollouts submitted their
pipelines, loaded both raw tables, executed the requested transformation, and
received terminal execution reward `1.0`. Tinker reported one completed batch,
`time/train_step = 3.47s`, and saved both training-state and sampler-weight
checkpoints. The Python process exited successfully.

This small fixture saturated: all four rewards were equal, so the batch does
not demonstrate a useful relative-reward learning signal or improvement in
model quality. Training should use a larger and more varied task set.

The upstream input generator produced 100 task bundles in a dry run with blank
account credentials. `books` parsed 15 raw tables and three targets. The
smaller official `trains` task has two raw tables (`cars`: 63 rows, `train`: 20
rows) and one target. Its public source files, task bundle, and private
Snowflake ground truth are staged locally for the credentialed run.

On 2026-09-28, Docker Desktop with Ubuntu WSL integration ran Airbyte Open
Source 2.3.0. The benchmark REST service returned its `cars` stream. The
Airbyte public API authenticated using local client credentials. Terraform
Airbyte provider 0.6.5 installed from a signed local mirror and created both
the declarative REST source and a local mirror of the benchmark's public CSV
file source. Airbyte discovery returned `cars` and `train` successfully. The
complete Terraform file for two sources, a Snowflake destination, and two
connections passed `terraform validate`; the destination and connections have
not yet been applied.

The reference `trains` SQL was run against the two public CSV inputs in DuckDB
and compared with official Snowflake ground truth: 20 rows, seven matching
columns, `TableScore(score=1.0)`. This checks the transformation logic without
claiming that Snowflake or Airbyte loading worked. The full project test suite
passed 11 tests on Windows and Ubuntu, including a test that Snowflake preview
queries use the task EL role rather than the administrator connection.

The credentialed Snowflake sync, official warehouse reward, and official
Tinker optimization step remain pending a locally configured Snowflake
administrator password. The Airbyte file source uses a local image containing
the benchmark's public CSV because this machine's Docker network encounters
certificate interception for the original Google Drive URL. The local
Terraform mirror avoids disabling certificate verification. No credentials or
Terraform state are published in this repository.
