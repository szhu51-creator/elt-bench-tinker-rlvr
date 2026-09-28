# Project plan and current status

This directory is the standalone ELT-Bench RLVR project. It contains the
environment and Tinker recipe in `src/elt_rlvr/`, a credential-free fixture in
`fixtures/`, tests in `tests/`, the implementation design in `DESIGN.md`, and
evidence in `VALIDATION.md`. The public GitHub repository is
https://github.com/szhu51-creator/elt-bench-tinker-rlvr.

## Completed

- [x] Stateful, tool-using ELT environment with a destination adapter boundary.
- [x] DuckDB end-to-end fixture and six passing integration tests.
- [x] Execution-derived reward with private ground truth and EL completion gates.
- [x] Tinker cookbook RL recipe; one live step on four local rollouts completed.
- [x] Design document and self-contained repository published.
- [x] Windows WSL 2 component and Docker Desktop installed on the development PC.

The four local rollout rewards were all `1.0`. That verifies the training
interface but gives no evidence of model improvement. The official Snowflake
path is implemented but still requires a credentialed rollout.

## After the Windows restart

1. Verify `wsl --version` and `wsl --list --verbose` in PowerShell. If Ubuntu
   is absent, install it with `wsl --install -d Ubuntu`, then launch it once to
   create a Linux user. The account owner chooses the Linux password locally.
2. Start Docker Desktop, select the WSL 2 backend, and verify that both the
   Docker client and server respond to `docker version`. The account owner
   reads and accepts Docker's service agreement on first launch.
3. Install Airbyte Open Source with `abctl`, start its local services, and
   import ELT-Bench's `setup/elt_snowflake.yaml` source manifest.
4. In the dedicated Snowflake trial account, create the benchmark role, user,
   and small auto-suspending warehouse from ELT-Bench's `setup/destination/setup.sql`.
   Replace the upstream example password with a unique secret before running
   the SQL. Store Snowflake and Airbyte credentials only in the official
   benchmark's local credential JSON files, never in this repository or chat.
5. Generate the official `books` input bundle, start the benchmark's source
   containers, and download Snowflake ground-truth CSVs. Review the upstream
   setup script before running it; the checked-out version contains an
   incomplete `unzip` line and may need a manual equivalent.
6. Build the benchmark execution image and run
   `elt-rlvr check-official --official-repo <path> --task-id books`.
7. Run a credentialed `books` rollout. Inspect the submitted Terraform/dbt
   artifacts and warehouse reward; fix integration issues and record the
   outcome in `VALIDATION.md`.
8. Run at least one Tinker step on the official task with execution rewards.
   Use varied tasks for any larger training experiment and compare held-out
   results before claiming improvement.

The project owner only needs to restart Windows, complete the first-launch
Ubuntu and Docker prompts, and enter account secrets locally. The agent can
handle installation, code changes, Airbyte/benchmark setup, execution,
debugging, tests, documentation, and publishing after those prompts.
