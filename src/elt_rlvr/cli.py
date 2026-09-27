"""Credential-free smoke run and deterministic action replay."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .episode import ELTEpisode
from .spec import TaskSpec


CORRECT_MINI_SQL = """SELECT c.customer_id, c.customer_name,
COUNT(o.order_id) AS order_count,
COALESCE(SUM(o.amount), 0) AS total_spend
FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.customer_name"""


def main() -> None:
    parser = argparse.ArgumentParser(description="ELT-Bench RLVR environment CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo-local", help="Run the credential-free end-to-end fixture")
    demo.add_argument("--run-root", type=Path, default=Path("run_artifacts"))
    replay = sub.add_parser("replay", help="Replay a JSON list of model tool actions")
    replay.add_argument("actions", type=Path)
    replay.add_argument("--run-root", type=Path, default=Path("run_artifacts"))
    replay.add_argument("--fixture-dir", type=Path)
    replay.add_argument("--official-repo", type=Path)
    replay.add_argument("--task-id", default="books")
    replay.add_argument("--inputs-dir", type=Path)
    replay.add_argument("--ground-truth-dir", type=Path)
    replay.add_argument("--credential-path", type=Path)
    check = sub.add_parser("check-official", help="Check official task files and Docker prerequisites")
    check.add_argument("--official-repo", type=Path, required=True)
    check.add_argument("--task-id", default="books")
    check.add_argument("--inputs-dir", type=Path)
    check.add_argument("--ground-truth-dir", type=Path)
    check.add_argument("--credential-path", type=Path)
    args = parser.parse_args()

    fixture = Path(__file__).resolve().parents[2] / "fixtures/mini_orders"
    if args.command == "check-official":
        spec = TaskSpec.official_snowflake(
            args.official_repo, args.task_id, inputs_dir=args.inputs_dir,
            ground_truth_dir=args.ground_truth_dir, credential_path=args.credential_path,
        )
        print(json.dumps({"task": spec.task_id, "raw_tables": len(spec.expected_raw_counts),
                          "targets": list(spec.target_tables), "docker": bool(shutil.which("docker"))}))
        return
    if args.command == "demo-local":
        spec = TaskSpec.local(fixture)
        actions = [
            {"tool": "inspect_task"},
            {"tool": "configure_el", "tables": ["customers", "orders"]},
            {"tool": "write_model", "model": "customer_totals", "sql": CORRECT_MINI_SQL},
            {"tool": "run_transforms"},
            {"tool": "submit_pipeline"},
        ]
    else:
        if args.official_repo:
            spec = TaskSpec.official_snowflake(
                args.official_repo, args.task_id, inputs_dir=args.inputs_dir,
                ground_truth_dir=args.ground_truth_dir, credential_path=args.credential_path,
            )
        else:
            spec = TaskSpec.local(args.fixture_dir or fixture)
        actions = json.loads(args.actions.read_text(encoding="utf-8"))
        if not isinstance(actions, list):
            raise ValueError("Action file must contain a JSON list")
    episode = ELTEpisode(spec, args.run_root)
    try:
        for item in actions:
            item = dict(item)
            name = item.pop("tool")
            result = episode.act(name, **item)
            print(json.dumps({"tool": name, "observation": result.observation,
                              "done": result.done, "reward": result.reward}))
            if result.done:
                break
    finally:
        episode.close()


if __name__ == "__main__":
    main()
