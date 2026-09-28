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
account credentials. The `books` task specification parsed its 15 raw tables
and three targets against downloaded public Snowflake ground-truth CSVs. This
checks task structure only; it does not prove extraction, transformation, or
grading on Snowflake. The official credentialed rollout still needs the
Snowflake account, source data, Docker, and Airbyte services; it was not run.
