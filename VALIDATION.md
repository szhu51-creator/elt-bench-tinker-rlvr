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

On 2026-09-28, the account owner approved a narrow database grant. The
administrator created `TRAINS` in Snowflake and granted `USAGE, CREATE SCHEMA`
on that database to `AIRBYTE_ROLE`. The official runtime then reset only
`TRAINS.AIRBYTE_SCHEMA` with the existing dedicated `AIRBYTE_USER`; no
administrator password was stored locally. The credentialed `trains` replay
successfully created the Snowflake destination and both Airbyte connections,
completed two real sync jobs, loaded `cars` and `train`, ran the dbt model, and
returned execution reward `1.0` (`raw_fraction=1.0`, `target/trains=1.0`).
The resulting target had 20 rows. The full test suite passed 14 tests after
adding coverage for delegated reset, role checks, and safe Terraform path
normalization.

Two live Tinker one-step optimization runs also used the official `trains`
environment and saved training/sampler checkpoints. The first used
`Qwen/Qwen3.5-4B`; both model rollouts exhausted 12 turns after repeatedly
passing path-qualified Terraform filenames. The tool now safely normalizes
those task-local names. The second used `Qwen/Qwen3.5-9B` with the fix and 16
turns. It wrote Terraform files, but its HCL did not pass the Airbyte provider
0.6.5 schema checks; neither model rollout submitted. Both official training
steps completed with execution reward `0.0` and are evidence of the real
optimization path, not evidence that these base models learned the task.
The separate deterministic official replay above verifies that a valid
end-to-end policy can earn reward `1.0`.

A structured Airbyte plan tool was then added for official tasks. It derives
the custom API and local CSV sources, Snowflake destination, and connections
from the task's public source names, while keeping credentials as references
to the per-rollout configuration. The generated `trains` plan passed
`terraform validate` with the real Airbyte provider 0.6.5. A second live
official replay using `configure_el(tables=["cars", "train"])` completed both
Airbyte syncs, dbt, and private warehouse grading with reward `1.0`. The full
suite passed 15 tests, including plan generation and source validation.

A new live Tinker one-step optimization used the structured tool on the same
official `trains` task with `Qwen/Qwen3.5-9B`, group size 2, and a 16-turn
budget. Both sampled trajectories completed EL, dbt, and terminal submission.
The group mean execution reward was `0.885714`; the raw-table fraction was
`1.0` and target-column fraction was `0.857143`. Tinker saved a training-state
and sampler checkpoint and reported successful completion. This shows a useful
positive official-task training signal, but the sampled policy did not reach
the full target score and one step does not establish learning or transfer.

This machine's Norton HTTPS scanner re-signs Snowflake certificates seen by
Docker and kind pods. The host's trusted public Norton root was exported to a
private local file, added to a private CA bundle for the ELT execution image,
and imported into a local derived Airbyte Snowflake connector image. TLS
verification remained enabled. The Airbyte file source uses a local image containing
the benchmark's public CSV because this machine's Docker network encounters
certificate interception for the original Google Drive URL. The local
Terraform mirror avoids disabling certificate verification. No credentials,
CA files, or Terraform state are published in this repository.
