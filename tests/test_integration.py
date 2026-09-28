from pathlib import Path

import pandas as pd
import json
import sys
from types import SimpleNamespace

import pytest

from elt_rlvr.backend import SnowflakeWarehouse
from elt_rlvr.airbyte_plan import render_snowflake_plan
from elt_rlvr.episode import ELTEpisode
from elt_rlvr.official import OfficialRuntime
from elt_rlvr.reward import compare_tables
from elt_rlvr.spec import TaskSpec


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/mini_orders"
CORRECT_SQL = """SELECT c.customer_id, c.customer_name,
COUNT(o.order_id) AS order_count,
COALESCE(SUM(o.amount), 0) AS total_spend
FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.customer_name"""


def test_end_to_end_local_rollout(tmp_path):
    episode = ELTEpisode(TaskSpec.local(FIXTURE), tmp_path)
    try:
        assert "customer_totals" in episode.act("inspect_task").observation
        assert "3" in episode.act("configure_el", tables=["customers", "orders"]).observation
        assert "staged" in episode.act("write_model", model="customer_totals", sql=CORRECT_SQL).observation
        assert "ok" in episode.act("run_transforms").observation
        assert "Ada" in episode.act("preview_sql", sql="SELECT * FROM customer_totals").observation
        final = episode.act("submit_pipeline")
        assert final.done and final.reward == 1.0
        assert episode.last_metrics["raw_fraction"] == 1.0
        assert episode.last_metrics["target/customer_totals"] == 1.0
    finally:
        episode.close()


def test_fabricated_final_table_without_el_scores_zero(tmp_path):
    episode = ELTEpisode(TaskSpec.local(FIXTURE), tmp_path)
    try:
        episode.act("write_model", model="customer_totals", sql="SELECT 1 AS customer_id, 'Ada' AS customer_name, 2 AS order_count, 15 AS total_spend")
        episode.act("run_transforms")
        assert episode.act("submit_pipeline").reward == 0.0
    finally:
        episode.close()


def test_execution_feedback_and_partial_reward(tmp_path):
    episode = ELTEpisode(TaskSpec.local(FIXTURE), tmp_path)
    try:
        episode.act("configure_el", tables=["customers", "orders"])
        episode.act("write_model", model="customer_totals", sql="SELECT customer_id, customer_name FROM customers")
        episode.act("run_transforms")
        score = episode.act("submit_pipeline").reward
        assert 0.2 < score < 1.0
    finally:
        episode.close()


def test_private_files_and_external_access_are_unavailable(tmp_path):
    episode = ELTEpisode(TaskSpec.local(FIXTURE), tmp_path)
    try:
        assert "outside" in episode.act("read_file", path="../private_ground_truth/customer_totals.csv").observation
        episode.act("configure_el", tables=["customers", "orders"])
        leak = episode.act("preview_sql", sql=f"SELECT * FROM read_csv('{FIXTURE / 'private_ground_truth/customer_totals.csv'}')")
        assert "permission" in leak.observation.lower() or "external" in leak.observation.lower()
    finally:
        episode.close()


def test_row_count_mismatch_cannot_earn_target_credit():
    gold = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
    actual = pd.DataFrame({"id": [1], "value": [10]})
    assert compare_tables(gold, actual).score == 0.0


def test_rollouts_have_isolated_warehouses(tmp_path):
    spec = TaskSpec.local(FIXTURE)
    first = ELTEpisode(spec, tmp_path)
    second = ELTEpisode(spec, tmp_path)
    try:
        first.act("configure_el", tables=["customers", "orders"])
        assert first.warehouse.count_raw("customers") == 3
        assert second.warehouse.count_raw("customers") is None
    finally:
        first.close()
        second.close()


def _official_spec(tmp_path):
    repo = tmp_path / "official"
    public_dir = repo / "inputs" / "books"
    (public_dir / "elt").mkdir(parents=True)
    (repo / "agents").mkdir()
    config = {
        "Airbyte": {"config": {"password": "airbyte-test-password"}},
        "snowflake": {"config": {
            "account": "account-test", "database": "books", "schema": "AIRBYTE_SCHEMA",
            "role": "AIRBYTE_ROLE", "warehouse": "AIRBYTE_WAREHOUSE",
            "username": "template-user", "password": "template-password",
        }},
    }
    import yaml

    config_path = public_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    credential_path = repo / "setup" / "destination" / "snowflake_credential.json"
    credential_path.parent.mkdir(parents=True)
    credential_path.write_text(json.dumps({"user": "reset-user"}), encoding="utf-8")
    return TaskSpec(
        task_id="books", destination="snowflake", public_dir=public_dir,
        expected_raw_counts={}, target_tables=(), ground_truth_dir=repo / "ground_truth",
        official_repo=repo, credential_path=credential_path,
    ), config_path


def _stub_official_prerequisites(monkeypatch):
    monkeypatch.setattr("elt_rlvr.official.shutil.which", lambda _name: "docker")
    monkeypatch.setitem(sys.modules, "common", SimpleNamespace(prepare_destination=lambda *_args: None))
    monkeypatch.setattr("elt_rlvr.official.OfficialRuntime._grant_database_usage", lambda _self: None)


def test_official_el_credentials_override_private_rollout_copy(tmp_path, monkeypatch):
    spec, public_config_path = _official_spec(tmp_path)
    public_config_before = public_config_path.read_text(encoding="utf-8")
    el_user = "rollout-el-user"
    el_password = "rollout-el-password-secret"
    monkeypatch.setenv("ELT_RLVR_SNOWFLAKE_EL_USER", el_user)
    monkeypatch.setenv("ELT_RLVR_SNOWFLAKE_EL_PASSWORD", el_password)
    _stub_official_prerequisites(monkeypatch)

    runtime = OfficialRuntime(spec, tmp_path / "runs")
    monkeypatch.setattr(runtime, "_command", lambda _argv, timeout: "container started")
    runtime.start()
    try:
        import yaml

        copied_config = yaml.safe_load((runtime.root / "config.yaml").read_text(encoding="utf-8"))
        snowflake_config = copied_config["snowflake"]["config"]
        assert snowflake_config["username"] == el_user
        assert snowflake_config["password"] == el_password
        profile = yaml.safe_load((runtime.root / "elt" / "profiles.yml").read_text(encoding="utf-8"))
        output = profile["elt_models"]["outputs"]["dev"]
        assert output["user"] == el_user
        assert output["password"] == el_password
        assert el_password in runtime._secrets
        assert public_config_path.read_text(encoding="utf-8") == public_config_before
        assert not (runtime.root / "snowflake_credential.json").exists()
    finally:
        runtime.started = False
        runtime.close()


def test_official_start_failure_cleans_copied_secrets_and_state(tmp_path, monkeypatch):
    spec, public_config_path = _official_spec(tmp_path)
    public_config_before = public_config_path.read_text(encoding="utf-8")
    monkeypatch.setenv("ELT_RLVR_SNOWFLAKE_EL_USER", "rollout-el-user")
    monkeypatch.setenv("ELT_RLVR_SNOWFLAKE_EL_PASSWORD", "rollout-el-password-secret")
    _stub_official_prerequisites(monkeypatch)
    runtime = OfficialRuntime(spec, tmp_path / "runs")

    def fail_after_creating_state(_argv, timeout):
        (runtime.root / "elt" / "terraform.tfstate").write_text("{}", encoding="utf-8")
        (runtime.root / "elt" / "terraform.tfstate.backup").write_text("{}", encoding="utf-8")
        raise RuntimeError("simulated container startup failure")

    monkeypatch.setattr(runtime, "_command", fail_after_creating_state)
    try:
        runtime.start()
        raise AssertionError("expected simulated startup failure")
    except RuntimeError as exc:
        assert "simulated container startup failure" in str(exc)

    assert public_config_path.read_text(encoding="utf-8") == public_config_before
    for sensitive_path in (
        runtime.root / "config.yaml",
        runtime.root / "elt" / "profiles.yml",
        runtime.root / "elt" / "terraform.tfstate",
        runtime.root / "elt" / "terraform.tfstate.backup",
    ):
        assert not sensitive_path.exists()


def test_official_close_cleans_secrets_when_docker_remove_fails(tmp_path, monkeypatch):
    spec, _ = _official_spec(tmp_path)
    _stub_official_prerequisites(monkeypatch)
    runtime = OfficialRuntime(spec, tmp_path / "runs")
    monkeypatch.setattr(runtime, "_command", lambda _argv, timeout: "container started")
    runtime.start()
    (runtime.root / "elt" / "terraform.tfstate").write_text("secret state", encoding="utf-8")

    def fail_remove(*_args, **_kwargs):
        raise RuntimeError("docker remove unavailable")

    monkeypatch.setattr("elt_rlvr.official.subprocess.run", fail_remove)
    with pytest.raises(RuntimeError, match="docker remove unavailable"):
        runtime.close()
    assert not runtime.started
    assert not (runtime.root / "config.yaml").exists()
    assert not (runtime.root / "elt" / "profiles.yml").exists()
    assert not (runtime.root / "elt" / "terraform.tfstate").exists()


def test_official_preview_does_not_use_admin_connection(tmp_path, monkeypatch):
    spec, _ = _official_spec(tmp_path)
    calls = []
    executed = []

    class FakeCursor:
        description = [("VALUE",)]

        def __init__(self, role):
            self.role = role

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, sql):
            executed.append((self.role, sql))

        def fetchall(self):
            return [(1,)]

    class FakeConnection:
        def __init__(self, role):
            self.role = role

        def cursor(self):
            return FakeCursor(self.role)

        def close(self):
            return None

    def connect(**kwargs):
        calls.append(kwargs)
        return FakeConnection(kwargs.get("role", "admin"))

    monkeypatch.setattr("snowflake.connector.connect", connect)
    warehouse = SnowflakeWarehouse(spec)
    try:
        warehouse.preview("SELECT 1")
        assert len(calls) == 2
        assert calls[1]["role"] == "AIRBYTE_ROLE"
        assert executed == [("AIRBYTE_ROLE", "SELECT * FROM (SELECT 1) q LIMIT 10")]
    finally:
        warehouse.close()


def test_official_reset_grants_only_task_database_usage(tmp_path, monkeypatch):
    spec, public_config_path = _official_spec(tmp_path)
    runtime = OfficialRuntime(spec, tmp_path / "runs")
    runtime.root.mkdir(parents=True)
    (runtime.root / "config.yaml").write_text(public_config_path.read_text())
    calls = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, sql):
            calls.append(sql)

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def cursor(self):
            return FakeCursor()

    monkeypatch.setattr("snowflake.connector.connect", lambda **_kwargs: FakeConnection())
    runtime._grant_database_usage()
    assert calls == ['GRANT USAGE ON DATABASE "BOOKS" TO ROLE "AIRBYTE_ROLE"']


def test_delegated_reset_uses_only_task_schema_and_el_role(tmp_path, monkeypatch):
    spec, _ = _official_spec(tmp_path)
    spec.credential_path.write_text(json.dumps({
        "account": "account-test", "user": "rollout-el-user", "password": "test-secret",
        "role": "AIRBYTE_ROLE", "warehouse": "AIRBYTE_WAREHOUSE",
    }), encoding="utf-8")
    monkeypatch.setenv("ELT_RLVR_SNOWFLAKE_RESET_MODE", "delegated")
    monkeypatch.setenv("ELT_RLVR_SNOWFLAKE_EL_USER", "rollout-el-user")
    monkeypatch.setenv("ELT_RLVR_SNOWFLAKE_EL_PASSWORD", "test-secret")
    monkeypatch.setattr("elt_rlvr.official.shutil.which", lambda _name: "docker")
    monkeypatch.setattr("elt_rlvr.official.OfficialRuntime._grant_database_usage",
                        lambda _self: pytest.fail("admin grant must not run"))
    statements = []

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def cursor(self):
            return self

        def execute(self, sql):
            statements.append(sql)

    def connect(**kwargs):
        assert kwargs["user"] == "rollout-el-user"
        assert kwargs["role"] == "AIRBYTE_ROLE"
        return FakeConnection()

    monkeypatch.setattr("snowflake.connector.connect", connect)
    runtime = OfficialRuntime(spec, tmp_path / "runs")
    monkeypatch.setattr(runtime, "_command", lambda _argv, timeout: "container started")
    try:
        runtime.start()
        assert statements == [
            'DROP SCHEMA IF EXISTS "BOOKS"."AIRBYTE_SCHEMA" CASCADE',
            'CREATE SCHEMA "BOOKS"."AIRBYTE_SCHEMA"',
        ]
    finally:
        runtime.started = False
        runtime.close()


def test_delegated_reset_rejects_wrong_role(tmp_path, monkeypatch):
    spec, public_config = _official_spec(tmp_path)
    spec.credential_path.write_text(json.dumps({
        "user": "template-user", "role": "ACCOUNTADMIN",
    }), encoding="utf-8")
    runtime = OfficialRuntime(spec, tmp_path / "runs")
    runtime.root.mkdir(parents=True)
    (runtime.root / "config.yaml").write_text(public_config.read_text(), encoding="utf-8")
    with pytest.raises(ValueError, match="task EL role"):
        runtime._reset_delegated_schema()


def test_terraform_tool_normalizes_task_paths_without_escape(tmp_path):
    spec, _ = _official_spec(tmp_path)
    runtime = OfficialRuntime(spec, tmp_path / "runs")
    (runtime.root / "elt").mkdir(parents=True)
    runtime.started = True
    try:
        assert runtime.write_terraform("_elt/traincars/src.tf", 'resource "airbyte_source_file" "x" {}') == "wrote elt/src.tf"
        assert (runtime.root / "elt/src.tf").is_file()
        with pytest.raises(ValueError, match="inside elt"):
            runtime.write_terraform("elt/../outside.tf", "")
        with pytest.raises(ValueError, match="main.tf"):
            runtime.write_terraform("elt/main.tf", "")
    finally:
        runtime.started = False
        runtime.close()


def test_structured_airbyte_plan_uses_declared_sources_and_config_references(tmp_path):
    from dataclasses import replace
    import yaml

    spec, config_path = _official_spec(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["custom_api"] = {"config": {"tables": ["cars"], "configuration": {}}}
    config["flat_files"] = [{"table": "train", "format": "csv", "path": "/local/train.csv"}]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    spec = replace(spec, expected_raw_counts={"cars": 63, "train": 20})
    plan = render_snowflake_plan(spec, ["cars", "train"], "run_123")
    assert 'resource "airbyte_source_custom" "api"' in plan
    assert 'resource "airbyte_source_file" "train"' in plan
    assert plan.count('resource "airbyte_connection"') == 2
    assert "local.cfg.snowflake.config.password" in plan
    assert "template-password" not in plan
    assert "airbyte-test-password" not in plan
    with pytest.raises(ValueError, match="task-declared"):
        render_snowflake_plan(spec, ["other"], "run_123")
    with pytest.raises(ValueError, match="without duplicates"):
        render_snowflake_plan(spec, ["cars", "cars"], "run_123")


def test_structured_airbyte_plan_supports_official_https_csv(tmp_path):
    from dataclasses import replace
    import yaml

    spec, config_path = _official_spec(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["flat_files"] = [{
        "table": "train", "format": "csv",
        "path": "https://drive.google.com/uc?export=download&id=public-file",
    }]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    spec = replace(spec, expected_raw_counts={"train": 20})
    plan = render_snowflake_plan(spec, ["train"], "run_123")
    assert "https_public_web = {}" in plan
    assert "url = local.cfg.flat_files[0].path" in plan
    assert "drive.google.com" not in plan

    config["flat_files"][0]["path"] = "http://insecure.example/train.csv"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="public HTTPS"):
        render_snowflake_plan(spec, ["train"], "run_123")
