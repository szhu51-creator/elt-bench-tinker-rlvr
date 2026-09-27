from pathlib import Path

import pandas as pd

from elt_rlvr.episode import ELTEpisode
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
