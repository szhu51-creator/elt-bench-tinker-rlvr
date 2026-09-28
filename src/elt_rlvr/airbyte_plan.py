"""Build a task-scoped Airbyte plan from public ELT-Bench source metadata."""

from __future__ import annotations

import json

import yaml

from .backend import valid_identifier
from .spec import TaskSpec


def render_snowflake_plan(spec: TaskSpec, selected: list[str], run_name: str) -> str:
    """Configure declared API, local CSV, or public HTTPS CSV streams."""
    if spec.destination != "snowflake":
        raise ValueError("This Airbyte plan requires a Snowflake task")
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Select at least one source table without duplicates")
    if not set(selected).issubset(spec.expected_raw_counts):
        raise ValueError("Only task-declared source tables may be selected")
    for name in selected:
        valid_identifier(name)
    config = yaml.safe_load((spec.public_dir / "config.yaml").read_text(encoding="utf-8"))
    api = config.get("custom_api", {}).get("config", {})
    api_names = set(api.get("tables", []))
    files = {item["table"]: (index, item) for index, item in enumerate(config.get("flat_files", []))}
    unsupported = set(selected) - api_names - set(files)
    if unsupported:
        raise ValueError(f"No supported public API/local CSV source for: {sorted(unsupported)}")
    providers: dict[str, str] = {}
    for name in selected:
        if name in files:
            entry = files[name][1]
            path = str(entry.get("path", ""))
            if entry.get("format") != "csv":
                raise ValueError(f"Only CSV file sources are supported: {name}")
            if path.startswith("/local/"):
                providers[name] = "local_filesystem_limited"
            elif path.startswith("https://"):
                providers[name] = "https_public_web"
            else:
                raise ValueError(f"Only local or public HTTPS CSV sources are supported: {name}")
    valid_identifier(spec.task_id)
    valid_identifier(run_name)
    prefix = f"elt-bench-{spec.task_id}-{run_name}"
    lines = [
        'locals { cfg = yamldecode(file("../config.yaml")) }',
        'provider "airbyte" {',
        '  server_url = local.cfg.Airbyte.config.server_url',
        '  client_id = local.cfg.Airbyte.config.client_id',
        '  client_secret = local.cfg.Airbyte.config.client_secret',
        '}',
        'resource "airbyte_destination_snowflake" "warehouse" {',
        f'  name = {json.dumps(prefix + "-snowflake")}',
        '  workspace_id = local.cfg.Airbyte.config.workspace_id',
        '  configuration = {',
        '    database = local.cfg.snowflake.config.database',
        '    host = "${local.cfg.snowflake.config.account}.snowflakecomputing.com"',
        '    role = local.cfg.snowflake.config.role',
        '    schema = local.cfg.snowflake.config.schema',
        '    username = local.cfg.snowflake.config.username',
        '    warehouse = local.cfg.snowflake.config.warehouse',
        '    credentials = { username_and_password = {',
        '      password = local.cfg.snowflake.config.password',
        '    } }',
        '  }',
        '}',
    ]
    api_selected = [name for name in selected if name in api_names]
    if api_selected:
        lines += [
            'resource "airbyte_source_custom" "api" {',
            f'  name = {json.dumps(prefix + "-api")}',
            '  workspace_id = local.cfg.Airbyte.config.workspace_id',
            '  definition_id = local.cfg.Airbyte.config.custom_api_definition_id',
            '  configuration = jsonencode(local.cfg.custom_api.config.configuration)',
            '}',
        ]
    for name in selected:
        if name not in files:
            continue
        index, _ = files[name]
        lines += [
            f'resource "airbyte_source_file" "{name}" {{',
            f'  name = {json.dumps(prefix + "-" + name)}',
            '  workspace_id = local.cfg.Airbyte.config.workspace_id',
            '  definition_id = local.cfg.Airbyte.config.files_definition_id',
            '  configuration = {',
            f'    dataset_name = {json.dumps(name)}',
            '    format = "csv"',
            f'    url = local.cfg.flat_files[{index}].path',
            f'    provider = {{ {providers[name]} = {{}} }}',
            '  }',
            '}',
        ]
    connection_groups = [("api", api_selected, "airbyte_source_custom.api.source_id")]
    connection_groups += [(name, [name], f"airbyte_source_file.{name}.source_id")
                          for name in selected if name in files]
    for label, streams, source_id in connection_groups:
        if not streams:
            continue
        lines += [
            f'resource "airbyte_connection" "{label}" {{',
            f'  name = {json.dumps(prefix + "-" + label + "-to-snowflake")}',
            f'  source_id = {source_id}',
            '  destination_id = airbyte_destination_snowflake.warehouse.destination_id',
            '  namespace_definition = "destination"',
            '  status = "active"',
            '  schedule = { schedule_type = "manual" }',
            '  configurations = { streams = [',
        ]
        lines += [f'    {{ name = {json.dumps(name)}, sync_mode = "full_refresh_overwrite" }}'
                  for name in streams]
        lines += ['  ] }', '}']
    return "\n".join(lines) + "\n"
