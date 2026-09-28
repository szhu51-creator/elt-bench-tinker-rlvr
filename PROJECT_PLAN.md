# Project plan and current status

This folder is the standalone ELT-Bench RLVR implementation. The public
repository is https://github.com/szhu51-creator/elt-bench-tinker-rlvr; the
design document is linked from `README.md`.

## Completed

- Stateful tool environment, destination adapter, execution-derived reward,
  and Tinker cookbook training recipe.
- Credential-free end-to-end fixture and 11 passing integration/security tests.
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

## Remaining verification

1. The account owner enters the Snowflake administrator password into the
   local `setup/destination/snowflake_credential.json`; it must never be pasted
   into chat or committed. Its `password` field is currently empty.
2. Reset the `trains` warehouse namespace and apply the validated Terraform
   plan, then run both Airbyte syncs and the dbt model in a real official
   rollout. Record the execution reward and repair any integration failures.
3. Run one Tinker optimization step using the official task's executed
   Snowflake reward. Update `VALIDATION.md` with the result.
4. Republish the final reviewed code and validation record to GitHub, and
   update the Google design document if the architecture changes.

The prepared benchmark sources and local Airbyte workspace survive ordinary
Docker Desktop restarts. The dedicated source service is configured to
restart automatically. If the user moves to a new machine, follow the official
ELT-Bench setup linked in `README.md` to recreate those external services.
