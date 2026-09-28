"""Destination boundary. Add a warehouse by implementing this small protocol."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

import pandas as pd
import yaml

from .spec import TaskSpec


IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def valid_identifier(name: str) -> str:
    if not IDENTIFIER.fullmatch(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


def select_only(sql: str) -> str:
    """Restrict model SQL to one SELECT; warehouse privileges remain the hard boundary."""
    text = sql.strip()
    if not text or ";" in text or not re.match(r"^(SELECT|WITH)\b", text, re.I):
        raise ValueError("Only one SELECT/WITH query without semicolons is allowed")
    return text


class Warehouse(Protocol):
    def count_raw(self, name: str) -> int | None: ...
    def fetch_target(self, name: str) -> pd.DataFrame | None: ...
    def preview(self, query: str, limit: int = 10) -> pd.DataFrame: ...
    def close(self) -> None: ...


class DuckDBWarehouse:
    """Credential-free analytical warehouse for local rollouts and tests."""

    def __init__(self, spec: TaskSpec, run_dir: Path):
        import duckdb

        self.spec = spec
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(run_dir / "warehouse.duckdb"))
        self.conn.execute("SET enable_external_access = false")

    def load(self, tables: list[str]) -> dict[str, int]:
        if self.spec.source_files is None:
            raise ValueError("Local source files are missing")
        unknown = set(tables) - set(self.spec.source_files)
        if unknown:
            raise ValueError(f"Unknown sources: {sorted(unknown)}")
        if not tables:
            raise ValueError("Select at least one source table")
        loaded = {}
        for name in tables:
            valid_identifier(name)
            path = self.spec.source_files[name]
            # Host-side loader reads only declared files. Agent SQL runs with
            # DuckDB external access disabled throughout the episode.
            frame = pd.read_csv(path)
            self.conn.register("__declared_source", frame)
            try:
                self.conn.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM __declared_source')
            finally:
                self.conn.unregister("__declared_source")
            loaded[name] = self.count_raw(name) or 0
        return loaded

    def run_models(self, models: dict[str, str]) -> dict[str, str]:
        results = {}
        for name, sql in models.items():
            valid_identifier(name)
            statement = select_only(sql)
            try:
                self.conn.execute(f'CREATE OR REPLACE TABLE "{name}" AS {statement}')
                results[name] = f"ok: {self.conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]} rows"
            except Exception as exc:
                results[name] = f"error: {str(exc)[:500]}"
        return results

    def count_raw(self, name: str) -> int | None:
        valid_identifier(name)
        try:
            return int(self.conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
        except Exception:
            return None

    def fetch_target(self, name: str) -> pd.DataFrame | None:
        valid_identifier(name)
        try:
            return self.conn.execute(f'SELECT * FROM "{name}"').df()
        except Exception:
            return None

    def preview(self, query: str, limit: int = 10) -> pd.DataFrame:
        query = select_only(query)
        limit = min(max(limit, 1), 20)
        return self.conn.execute(f"SELECT * FROM ({query}) AS q LIMIT {limit}").df()

    def close(self) -> None:
        self.conn.close()


class SnowflakeWarehouse:
    """Official ELT-Bench namespace and evaluation query mapping."""

    def __init__(self, spec: TaskSpec):
        import json
        import snowflake.connector

        if not spec.credential_path or not spec.official_repo:
            raise ValueError("Official Snowflake paths are required")
        self.spec = spec
        cfg = json.loads(spec.credential_path.read_text(encoding="utf-8"))
        self.conn = snowflake.connector.connect(**cfg)
        warehouse = yaml.safe_load((spec.public_dir / "config.yaml").read_text(encoding="utf-8"))["snowflake"]["config"]
        try:
            self.preview_conn = snowflake.connector.connect(
                account=warehouse["account"],
                user=warehouse.get("user") or warehouse["username"],
                password=warehouse["password"],
                role=warehouse["role"],
                warehouse=warehouse["warehouse"],
                database=warehouse["database"],
                schema=warehouse["schema"],
            )
        except BaseException:
            self.conn.close()
            raise

    def _query(self, sql: str, *, preview: bool = False) -> pd.DataFrame:
        connection = self.preview_conn if preview else self.conn
        with connection.cursor() as cur:
            cur.execute(sql)
            return pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])

    def count_raw(self, name: str) -> int | None:
        valid_identifier(name)
        try:
            df = self._query(f"SELECT COUNT(*) AS N FROM {valid_identifier(self.spec.task_id)}.AIRBYTE_SCHEMA.{name}")
            return int(df.iloc[0, 0])
        except Exception:
            return None

    def fetch_target(self, name: str) -> pd.DataFrame | None:
        valid_identifier(name)
        sql_file = self.spec.official_repo / "evaluation/sql" / self.spec.task_id / f"{name}.sql"
        try:
            if sql_file.is_file():
                sql = sql_file.read_text(encoding="utf-8")
                sql = re.sub(
                    r"\b(FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)",
                    lambda m: f"{m.group(1)} {m.group(2)}.AIRBYTE_SCHEMA.{m.group(3)}", sql,
                    flags=re.I,
                )
            else:
                sql = f"SELECT * FROM {valid_identifier(self.spec.task_id)}.AIRBYTE_SCHEMA.{name}"
            return self._query(sql)
        except Exception:
            return None

    def preview(self, query: str, limit: int = 10) -> pd.DataFrame:
        query = select_only(query)
        # Read-only role grants must restrict access to the current task namespace.
        limit = min(max(limit, 1), 20)
        return self._query(f"SELECT * FROM ({query}) q LIMIT {limit}", preview=True)

    def close(self) -> None:
        self.preview_conn.close()
        self.conn.close()
