# ELT-Bench Tinker RLVR

A stateful, tool-using ELT environment and Tinker cookbook RL recipe. Each rollout
inspects an ELT task, performs extraction/loading, writes transformation SQL,
uses execution feedback, and submits a warehouse state for private grading.

[Concise implementation design (Google Docs)](https://docs.google.com/document/d/1WDJoBACmtEmbLlaGtROHIn-Tj_diEmJif3NKO9BR-nY/edit)

The credential-free path uses DuckDB and a small CSV fixture. The official path
targets Snowflake using ELT-Bench's generated task bundles, Terraform/Airbyte
runtime, dbt, and destination reset helper. Warehouse access is isolated behind
`backend.py`; a new destination adds an adapter and an execution runtime rather
than changing the rollout or reward logic.

## Local integration test

Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate               # Windows: .venv\Scripts\Activate.ps1
pip install -e '.[test]'
pytest -q
elt-rlvr demo-local
```

`demo-local` runs a complete ELT pipeline and should end with `reward: 1.0`.
The fixture is in `fixtures/mini_orders`. You can replay a list of tool actions:

```bash
elt-rlvr replay actions.json
```

Each action is a JSON object with a `tool` key and the tool arguments, for
example `{"tool":"configure_el","tables":["customers","orders"]}`. The tool
names are `inspect_task`, `read_file`, `configure_el`, `run_el`, `write_model`,
`run_transforms`, `preview_sql` (local), `preview_table` (official), and
`submit_pipeline`. `configure_el` loads
task-declared CSVs in DuckDB; on official Snowflake tasks it writes a new
Airbyte Terraform file with `filename` and `content`, then `run_el` applies
Terraform and waits for Airbyte syncs.

## One Tinker RL optimization step

Install the optional training dependencies, authenticate with `tinker auth
login` (or set `TINKER_API_KEY`), and run:

```bash
pip install -e '.[train,test]'
python -m elt_rlvr.tinker_recipe --max-steps 1 --group-size 4
```

Tinker requires the organization to have an available balance or an enterprise
agreement before it accepts training requests; see the
[official billing guidance](https://tinker-docs.thinkingmachines.ai/tinker/data-model/).

This reuses `tinker_cookbook.tool_use.build_agent_tool_env`,
`EnvGroupBuilder`, `RLDatasetBuilder`, and `tinker_cookbook.rl.train.main`.
The model's tool calls change a per-rollout warehouse. The terminal reward
comes from executing queries against that warehouse and comparing with private
CSV ground truth. A group size greater than one permits reward-relative
advantages. The recipe uses the cookbook's agentic rollout policy and LoRA
training. A full training run increases `--max-steps` and supplies more task
groups; the bundled local fixture is deliberately small and intended as a
smoke test.

## Official ELT-Bench Snowflake rollout

These prerequisites follow the [official ELT-Bench setup](https://github.com/uiuc-kang-lab/ELT-Bench):

1. Clone the ELT-Bench repository; set up Docker, Airbyte, source containers,
   and a Snowflake destination as described there.
2. Populate `setup/airbyte/airbyte_credential.json` and
   `setup/destination/snowflake_credential.json` outside this repository. For
   Airbyte 2.x, put its `client_id` and `client_secret` in the generated
   `Airbyte.config`; the runtime obtains and refreshes bearer tokens.
3. Generate inputs with `python setup/write_config.py --destination snowflake`.
4. Download the public `gt_snowflake/**` dataset under
   `<ELT-Bench>/ground_truth/` using the official README command.
5. Build the official execution image:
   `docker build -t elt-swe agents/SWE-agent/docker/elt-swe`.
6. Ensure the benchmark's `elt-docker_elt_network` Docker network and Airbyte
   services are running.

Set `ELT_RLVR_SNOWFLAKE_EL_USER` and `ELT_RLVR_SNOWFLAKE_EL_PASSWORD` in the
host environment to provide the Snowflake account used by Airbyte and dbt.
Set both variables together. They override only the per-rollout copy of
`config.yaml`; the official input bundle and host-only reset/grader credential
file remain separate.

Install the warehouse connector and check a task:

```bash
pip install -e '.[train,snowflake,test]'
elt-rlvr check-official --official-repo /path/to/ELT-Bench --task-id trains
```

Run one Tinker optimization step on the official task:

```bash
python -m elt_rlvr.tinker_recipe \
  --official-repo /path/to/ELT-Bench \
  --task-id trains --group-size 2 --max-steps 1 \
  --run-root run_artifacts/official
```

The official run resets **only the selected task's Snowflake namespace** before
each rollout, exactly as ELT-Bench's `agents/common.prepare_destination` does.
Rollouts for one task are serialized to prevent their resets, Airbyte jobs,
and grading from interfering. Only generated task inputs enter the Docker
container; ground truth and the host credential JSON stay outside. The
container can write only the task workspace. The model can write an Airbyte
Terraform file and target dbt model SQL, run the fixed Terraform/Airbyte/dbt
commands, inspect read-only warehouse queries, then submit.

The Snowflake reset uses a host-only administrative credential and grants the
task database's `USAGE` privilege to its EL role. Warehouse previews use that
EL role, while private grading uses the host-only connection. Configure the
EL role with access only to the task's sources and destination. Airbyte 0.6.5
provider operations are applied with `-parallelism=1` because concurrent
credential-cache refresh can crash the provider.

For a Snowflake account where the administrator is signed in only through
Snowsight, a delegated reset is also supported. An administrator first creates
the task database and grants its EL role `USAGE, CREATE SCHEMA` on that database.
Set `ELT_RLVR_SNOWFLAKE_RESET_MODE=delegated` and pass a private JSON credential
file for that EL user with `--credential-path`. The runtime then drops and
recreates only the task schema as that role, and private grading connects with
the same role. The JSON user and role must match the task configuration. This
mode does not require a locally stored administrator password; keep the EL
credential file outside the repository.

```bash
export ELT_RLVR_SNOWFLAKE_RESET_MODE=delegated
elt-rlvr check-official --official-repo /path/to/ELT-Bench \
  --task-id trains --credential-path /private/snowflake_el.json
python -m elt_rlvr.tinker_recipe --official-repo /path/to/ELT-Bench \
  --task-id trains --credential-path /private/snowflake_el.json \
  --group-size 2 --max-steps 1 --run-root run_artifacts/official
```

If the Docker container cannot verify `registry.terraform.io` because of a
local TLS proxy, create a signed provider mirror from a trusted Linux host with
`terraform providers mirror /path/to/mirror` in a directory containing the
benchmark's `main.tf`, then set
`ELT_RLVR_TERRAFORM_MIRROR=/path/to/mirror` before starting rollouts.
If an HTTPS inspecting antivirus or proxy re-signs Snowflake certificates for
Docker containers, export its trusted public root certificate, append it to a
normal CA bundle, and set `ELT_RLVR_CA_BUNDLE=/path/to/ca-bundle.pem`. The
runtime mounts this bundle read-only and retains TLS certificate validation.

Use Terraform's `yamldecode(file("../config.yaml"))` to reference the generated
Airbyte and Snowflake values. Literal credentials, provisioners, modules,
external data sources, and outputs are rejected by the tool. The model's
`read_file` view of `config.yaml` redacts secret values. Host-side run files
containing credentials or Terraform state are removed when the episode closes;
the submitted HCL and model SQL remain for review.

## Reward and observations

The episode is stateful: file writes, warehouse loads, sync jobs, and dbt runs
persist across tool turns. `submit_pipeline` is terminal; an unsubmitted
episode receives zero. Execution errors return to the model for repair without
granting reward. At submission, the grader checks expected source row counts
and requires a successful EL action. When that gate passes, it compares each
requested target table with private ground truth, including exact row count,
column presence, and values (numeric tolerance `rtol=1e-2`, `atol=1e-9`,
case-insensitive trimmed strings). The score is:

```text
EL incomplete: 0.2 × fraction of raw tables with correct row counts, only if EL ran
EL complete:   0.2 + 0.8 × mean(target column match fractions), only if transforms ran
```

All target credit requires complete EL. The reward uses warehouse results,
not a model claim, artifact existence, or tool-call count. The official
Snowflake adapter uses the benchmark's `evaluation/sql/<task>/<table>.sql`
when present, as the official evaluator does.

## Extension point

Implement `Warehouse.count_raw`, `fetch_target`, `preview`, and `close` for a
new destination; add a runtime that resets its namespace and executes EL/dbt.
The `TaskSpec` loader maps the destination's generated input bundle and ground
truth. `ELTEpisode` and the Tinker bridge stay unchanged.

## Limits

The local DuckDB test, official `trains` Airbyte/Snowflake/dbt replay with
reward `1.0`, and one-step Tinker optimizations on both local and official
tasks have run; see [VALIDATION.md](VALIDATION.md) for results. The official
4B and 9B sampled policies received zero reward, so model quality remains a
training limitation. The Terraform text filter narrows
the model's tool surface but is not a substitute for a restricted Snowflake
role, isolated Airbyte account, or Docker/network policy in a production
training deployment. Train with benchmark tasks that are disjoint from the
evaluation set to reduce answer contamination.

See [DESIGN.md](DESIGN.md) for the concise implementation design.
See [PROJECT_PLAN.md](PROJECT_PLAN.md) for current progress and the Windows
restart-to-official-rollout checklist.
