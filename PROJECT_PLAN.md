# Project plan and current status

This folder is the standalone ELT-Bench RLVR implementation. The public
repository is https://github.com/szhu51-creator/elt-bench-tinker-rlvr; the
design document is linked from `README.md`.

## Completed

- Stateful tool environment, destination adapter, execution-derived reward,
  and Tinker cookbook training recipe.
- Credential-free end-to-end fixture and 15 passing integration/security tests.
- One live Tinker optimizer step with four local fixture rollouts, each reward
  `1.0`. The equal rewards do not establish model improvement.
- Windows Docker Desktop, Ubuntu WSL integration, Airbyte 2.3.0, and the
  benchmark REST source installed on the development machine.
- Official `trains` task bundle staged, public data acquired, and both Airbyte
  source streams (`cars`, `train`) discovered through real connector jobs.
- Signed Terraform provider mirror prepared for this machine's intercepted
  Docker TLS path. Complete source/destination/connection HCL validates.
- Reference transformation SQL matches the official `trains` ground truth
  offline, 20 rows and seven columns.
- A dedicated Snowflake EL user, role, and small auto-suspending warehouse were
  created. The EL user's password is stored locally outside the public repo.
- The administrator created only the `TRAINS` test database and granted the EL
  role `USAGE, CREATE SCHEMA` on it. The delegated reset needs no locally
  stored administrator password.
- A credentialed official `trains` replay completed Airbyte syncs, dbt, and
  warehouse grading with execution reward `1.0`.
- A structured source-selection tool generated valid Airbyte HCL for the
  official task and completed a second live replay with reward `1.0`.
- Two one-step Tinker runs on the official task completed and saved checkpoints.
  The sampled 4B and 9B policies earned zero because their tool trajectories
  failed to complete valid Terraform and submission within the turn budget.
- With the structured tool, a further official 9B Tinker step completed two
  submitted trajectories with mean execution reward `0.885714` and saved
  checkpoints. The target score was `0.857143` on average.

## Remaining research

For a stronger research result, train on multiple disjoint official tasks and
evaluate held-out full-pipeline success. This needs more task-specific source
support, training time, and compute credits. The three specified validation
paths have been executed, but one training step does not establish a general
ELT-solving policy.

The prepared benchmark sources and local Airbyte workspace survive ordinary
Docker Desktop restarts. The dedicated source service is configured to
restart automatically. If the user moves to a new machine, follow the official
ELT-Bench setup linked in `README.md` to recreate those external services.
