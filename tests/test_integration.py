from pathlib import Path

import pandas as pd
import json
import sys
from types import SimpleNamespace

import pytest

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
