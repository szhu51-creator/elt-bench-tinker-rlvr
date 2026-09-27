"""One stateful rollout, its tool transitions, and terminal reward."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .backend import DuckDBWarehouse, SnowflakeWarehouse, Warehouse, select_only
from .official import OfficialRuntime
from .reward import compare_tables
from .spec import TaskSpec


@dataclass(frozen=True)
class Transition:
    observation: str
    done: bool = False
    reward: float = 0.0


class ELTEpisode:
    def __init__(self, spec: TaskSpec, run_root: Path, *, image: str = "elt-swe", network: str = "elt-docker_elt_network"):
        self.spec = spec
        self.run_dir = run_root.resolve() / f"rollout-{uuid.uuid4().hex[:12]}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.runtime: OfficialRuntime | None = None
        if spec.destination == "duckdb":
            self.warehouse: Warehouse = DuckDBWarehouse(spec, self.run_dir)
        elif spec.destination == "snowflake":
            self.runtime = OfficialRuntime(spec, self.run_dir, image=image, network=network)
            try:
                self.runtime.start()
                self.warehouse = SnowflakeWarehouse(spec)
            except BaseException:
                self.runtime.close()
                raise
        else:
            raise ValueError(f"Unsupported destination: {spec.destination}")
        self.models: dict[str, str] = {}
        self.el_succeeded = False
        self.transform_ran = False
        self.submitted = False
        self.last_score = 0.0
        self.last_metrics: dict[str, float] = {}

    def _read_public(self, rel: str) -> str:
        public = self.spec.public_dir.resolve()
        path = (public / rel).resolve()
        if not path.is_relative_to(public) or not path.is_file():
            raise ValueError("File is outside the public task bundle")
        if rel.endswith("_credential.json") or rel.endswith("profiles.yml") or rel.endswith(".tfstate"):
            raise ValueError("Credential and state files are private")
        if path.stat().st_size > 100_000:
            raise ValueError("File is too large to inspect")
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.name == "config.yaml":
            cfg = yaml.safe_load(text)
            def redact(obj: Any) -> Any:
                if isinstance(obj, dict):
                    return {k: ("[configured]" if any(s in k.lower() for s in ("password", "secret", "token", "access_key")) else redact(v)) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [redact(x) for x in obj]
                return obj
            return yaml.safe_dump(redact(cfg), sort_keys=False)
        return text

    def act(self, name: str, **kwargs: Any) -> Transition:
        if self.submitted:
            return Transition("Episode already submitted", True, self.last_score)
        try:
            if name == "inspect_task":
                return Transition(self.spec.public_summary())
            if name == "read_file":
                return Transition(self._read_public(str(kwargs["path"])))
            if name == "configure_el":
                if self.spec.destination == "duckdb":
                    tables = kwargs["tables"]
                    if not isinstance(tables, list) or not all(isinstance(x, str) for x in tables):
                        raise ValueError("tables must be a list of source names")
                    loaded = self.warehouse.load(tables)  # type: ignore[attr-defined]
                    self.el_succeeded = set(self.spec.expected_raw_counts).issubset(loaded)
                    return Transition(json.dumps({"loaded": loaded}))
                assert self.runtime is not None
                return Transition(self.runtime.write_terraform(str(kwargs["filename"]), str(kwargs["content"])))
            if name == "run_el":
                if self.runtime is None:
                    return Transition("Local CSV sources load during configure_el")
                result = self.runtime.run_extract_load()
                self.el_succeeded = True
                return Transition(result)
            if name == "write_model":
                model = str(kwargs["model"])
                if model not in self.spec.target_tables:
                    raise ValueError(f"Unknown target model: {model}")
                sql = select_only(str(kwargs["sql"]))
                self.models[model] = sql
                if self.runtime:
                    return Transition(self.runtime.write_model(model, sql))
                return Transition(f"staged model {model}")
            if name == "run_transforms":
                if self.runtime:
                    result = self.runtime.run_transforms()
                else:
                    result = json.dumps(self.warehouse.run_models(self.models))  # type: ignore[attr-defined]
                self.transform_ran = True
                return Transition(result)
            if name == "preview_sql":
                if self.spec.destination != "duckdb":
                    raise ValueError("Use preview_table for official warehouse tasks")
                frame = self.warehouse.preview(str(kwargs["sql"]))
                return Transition(frame.to_json(orient="records", date_format="iso")[:4000])
            if name == "preview_table":
                if self.spec.destination != "snowflake":
                    raise ValueError("Use preview_sql for local tasks")
                table = str(kwargs["table"])
                if table not in self.spec.expected_raw_counts and table not in self.spec.target_tables:
                    raise ValueError("Table is outside the current task")
                frame = self.warehouse.preview(
                    f"SELECT * FROM {self.spec.task_id}.AIRBYTE_SCHEMA.{table}"
                )
                return Transition(frame.to_json(orient="records", date_format="iso")[:4000])
            if name == "submit_pipeline":
                self.submitted = True
                self.last_score, self.last_metrics = self.grade()
                return Transition(json.dumps({"reward": self.last_score, "metrics": self.last_metrics}), True, self.last_score)
            raise ValueError(f"Unknown tool: {name}")
        except Exception as exc:
            return Transition(f"{type(exc).__name__}: {str(exc)[:1200]}")

    def grade(self) -> tuple[float, dict[str, float]]:
        if not self.submitted:
            return 0.0, {"submitted": 0.0}
        raw = {name: self.warehouse.count_raw(name) for name in self.spec.expected_raw_counts}
        raw_matches = sum(raw[name] == expected for name, expected in self.spec.expected_raw_counts.items())
        raw_fraction = raw_matches / max(len(raw), 1)
        el_complete = self.el_succeeded and raw_fraction == 1.0
        metrics = {"submitted": 1.0, "el_success": float(self.el_succeeded), "raw_fraction": raw_fraction}
        if not el_complete:
            # A fabricated target table cannot bypass extraction/loading.
            return 0.2 * raw_fraction if self.el_succeeded else 0.0, metrics
        scores = []
        keys_by_table: dict[str, list[str]] = {}
        if self.spec.official_repo:
            sort_keys = json.loads((self.spec.official_repo / "evaluation/sort_key.json").read_text(encoding="utf-8"))
            keys_by_table = sort_keys.get(self.spec.task_id, {})
        for table in self.spec.target_tables:
            gold_path = self.spec.ground_truth_dir / f"{table}.csv"
            actual = self.warehouse.fetch_target(table)
            if not gold_path.is_file() or actual is None:
                scores.append(0.0)
                metrics[f"target/{table}"] = 0.0
                continue
            gold = pd.read_csv(gold_path, dtype=str)
            result = compare_tables(gold, actual, keys_by_table.get(table, []))
            scores.append(result.score)
            metrics[f"target/{table}"] = result.score
        target_fraction = sum(scores) / max(len(scores), 1)
        metrics["target_fraction"] = target_fraction
        reward = 0.2 + 0.8 * target_fraction if self.transform_ran else 0.2
        return round(reward, 6), metrics

    def close(self) -> None:
        try:
            self.warehouse.close()
        finally:
            if self.runtime:
                self.runtime.close()
