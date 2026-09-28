# Implementation design — ELT-Bench RLVR

## Task and episode

An ELT-Bench task defines heterogeneous extraction sources and stream names,
an Airbyte destination, a target warehouse namespace, and dbt target models
described in `data_model.yaml`. The agent must configure connections and sync
jobs, produce the target SQL, run it, inspect errors, and submit. A rollout is
**stateful**: Terraform files, Airbyte connections/jobs, the warehouse, dbt
models, and execution logs change after each action. A fresh rollout gets a
fresh execution container and reset warehouse namespace. Airbyte resources
live in a shared local workspace; rollouts for a task are serialized.

The initial prompt exposes the task's source names and model descriptions.
Tools expose bounded reads of the generated input bundle, writes to task-local
Terraform files or target model SQL, fixed execution commands, and read-only
SQL previews. `submit_pipeline` or the rollout budget ends the episode.
The model never receives private ground-truth CSVs or grading queries.

## Components

| Component | Responsibility |
| --- | --- |
| `TaskSpec` | Loads a local fixture or generated official Snowflake bundle; separates public inputs and private grading data. |
| `ELTEpisode` | Owns one mutable rollout, validates tool actions, records successful EL and dbt execution, and handles termination. |
| `airbyte_plan.py` | Compiles declared custom API and local CSV streams into a task-scoped Airbyte Terraform plan with credential references. |
| `DuckDBWarehouse` | Local OLAP destination, task-declared CSV extraction/loading, SELECT model materialization, external file access disabled. |
| `OfficialRuntime` + `SnowflakeWarehouse` | Uses the benchmark reset helper or a delegated task-schema reset, Docker execution image, Terraform/Airbyte/dbt, and destination-specific query mapping. |
| `reward.py` | Executes warehouse queries and compares them to private target CSVs with exact row counts and ELT-Bench-compatible value tolerance. |
| `tinker_recipe.py` | Reuses cookbook tool environment, grouped rollouts, dataset builder, and RL training loop. |

## Reward

At terminal submission, let `r_raw` be the fraction of expected raw tables
present with exact row counts. An explicit successful extraction/loading
action is required. If any raw table is wrong, reward is at most `0.2 × r_raw`.
If all are correct, compare every final target against private ground truth:
rows must match in count, expected columns must exist, and numeric/string
values must match after deterministic sorting. Each table gets a fraction of
matching expected columns; average across targets to get `r_target`. After
running transforms, reward is `0.2 + 0.8 × r_target`. An unsubmitted rollout
scores zero. This is an outcome reward with dense execution-derived terminal
partial credit. Error messages are process feedback, not reward, because
granting points for tool use would incentivize empty or redundant actions.

## Integrity and efficiency

- Ground truth and the host credential file are absent from the model's tools
  and Docker mount. The model's config view redacts secrets. The local DuckDB
  connection denies external file access after host-side source loading.
  Snowflake previews use the task's EL role; private grading uses a host-only
  connection. The administrator reset mode uses an administrator credential;
  delegated mode uses the EL role after a one-time database grant.
- The model cannot run an arbitrary shell command. It can only write bounded
  Airbyte Terraform and SELECT dbt model text, then invoke fixed commands.
  For supported Snowflake sources, it selects stream names and the environment
  generates the Airbyte plan, avoiding provider-schema guesswork. Unsupported
  source types can use the bounded raw-HCL fallback.
  The Terraform tool rejects provisioners, modules, external resources,
  outputs, and literal configured credentials. A least-privilege warehouse
  role and network isolation remain required defense in depth.
- EL success is recorded from actual sync/load execution, then independently
  checked through raw warehouse row counts. Target credit is gated on EL, so
  fabricating a final table alone cannot earn it.
- Local rollouts use separate DuckDB files. Official rollouts of one Snowflake
  task are serialized because the benchmark resets a shared task namespace.
  Terraform provider downloads use a shared plugin cache or signed local mirror;
  Terraform apply is serialized to avoid the provider's credential-cache race.
  Airbyte 2.3 uses client-credential tokens for sync requests. Grading runs once
  at terminal submission, not after every tool call. Warehouse previews are
  capped at 20 rows and do not reveal ground truth.

## Verification and extension

The credential-free integration test loads CSV sources, runs a transformation,
submits, and checks reward `1.0`. Negative tests cover fabricated final
tables, missing columns, row-count mismatch, private-file access, and rollout
isolation. A credentialed `trains` rollout completed both Airbyte syncs, dbt,
and private Snowflake grading with reward `1.0`. A one-step Tinker command uses
the same environment and execution-derived reward. Adding another destination
requires a `Warehouse` adapter, task loader, and destination runtime; the
episode and reward code do not change.
