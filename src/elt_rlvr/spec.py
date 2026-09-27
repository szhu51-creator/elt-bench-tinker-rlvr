"""Public task metadata and private grading inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    destination: str
    public_dir: Path
    expected_raw_counts: dict[str, int]
    target_tables: tuple[str, ...]
    ground_truth_dir: Path
    source_files: dict[str, Path] | None = None
    official_repo: Path | None = None
    credential_path: Path | None = None

    @classmethod
    def local(cls, fixture_dir: Path) -> "TaskSpec":
        fixture_dir = fixture_dir.resolve()
        data = json.loads((fixture_dir / "task.json").read_text(encoding="utf-8"))
        public = fixture_dir / "public"
        sources = {name: public / rel for name, rel in data["sources"].items()}
        for file in sources.values():
            if not file.is_file() or not file.resolve().is_relative_to(public.resolve()):
                raise ValueError(f"Invalid source file: {file}")
        return cls(
            task_id=data["task_id"], destination="duckdb", public_dir=public,
            expected_raw_counts={k: int(v) for k, v in data["raw_counts"].items()},
            target_tables=tuple(data["targets"]),
            ground_truth_dir=fixture_dir / "private_ground_truth",
            source_files=sources,
        )

    @classmethod
    def official_snowflake(
        cls, official_repo: Path, task_id: str, *, inputs_dir: Path | None = None,
        ground_truth_dir: Path | None = None, credential_path: Path | None = None,
    ) -> "TaskSpec":
        official_repo = official_repo.resolve()
        inputs_dir = (inputs_dir or official_repo / "inputs").resolve()
        public = inputs_dir / task_id
        if not public.is_dir() or not public.is_relative_to(inputs_dir):
            raise ValueError(f"Generated Snowflake input bundle missing: {public}")
        if not (public / "config.yaml").is_file():
            raise ValueError("Run the official setup/write_config.py first")
        model = yaml.safe_load((public / "data_model.yaml").read_text(encoding="utf-8"))
        counts = json.loads((official_repo / "evaluation/table.json").read_text(encoding="utf-8"))
        if task_id not in counts:
            raise ValueError(f"Unknown official task: {task_id}")
        gt = (ground_truth_dir or official_repo / "ground_truth/gt_snowflake").resolve() / task_id
        if not gt.is_dir():
            raise ValueError(f"Ground truth missing for {task_id}: {gt}")
        credential = (credential_path or official_repo / "setup/destination/snowflake_credential.json").resolve()
        if not credential.is_file():
            raise ValueError(f"Snowflake credential missing: {credential}")
        return cls(
            task_id=task_id, destination="snowflake", public_dir=public,
            expected_raw_counts={k: int(v) for k, v in counts[task_id].items()},
            target_tables=tuple(item["name"] for item in model["models"]),
            ground_truth_dir=gt, official_repo=official_repo,
            credential_path=credential,
        )

    def public_summary(self) -> str:
        model_path = self.public_dir / "data_model.yaml"
        model = model_path.read_text(encoding="utf-8") if model_path.is_file() else ""
        return (
            f"Task: {self.task_id}\nDestination: {self.destination}\n"
            f"Raw tables: {', '.join(self.expected_raw_counts)}\n"
            f"Required final models: {', '.join(self.target_tables)}\n"
            f"Data model:\n{model[:12000]}"
        )
