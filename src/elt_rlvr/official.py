"""Official Snowflake task runtime using ELT-Bench's own Airbyte/Terraform/dbt image."""

from __future__ import annotations

import shutil
import re
import os
import json
import subprocess
import sys
import uuid
from pathlib import Path

import yaml

from .spec import TaskSpec


class OfficialRuntime:
    def __init__(
        self, spec: TaskSpec, run_root: Path, *, image: str = "elt-swe",
        network: str = "elt-docker_elt_network",
    ):
        if spec.destination != "snowflake" or spec.official_repo is None:
            raise ValueError("OfficialRuntime currently supports Snowflake")
        self.spec = spec
        self.root = run_root.resolve() / f"{spec.task_id}-{uuid.uuid4().hex[:10]}"
        self.image = image
        self.network = network
        self.container = f"elt-rlvr-{uuid.uuid4().hex[:12]}"
        self.started = False
        self._secrets: list[str] = []

    def start(self) -> None:
        if not shutil.which("docker"):
            raise RuntimeError("Docker is required for official ELT-Bench rollouts")
        if not self.spec.credential_path:
            raise ValueError("A Snowflake credential path is required")
        self.root.mkdir(parents=True, exist_ok=False)
        try:
            for item in self.spec.public_dir.iterdir():
                if item.name.endswith("_credential.json"):
                    continue
                dest = self.root / item.name
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
            self._apply_el_credentials()
            shutil.copy2(Path(__file__).with_name("airbyte_sync.py"), self.root / "rlvr_airbyte_sync.py")
            self._write_dbt_scaffold()
            if os.environ.get("ELT_RLVR_SNOWFLAKE_RESET_MODE") == "delegated":
                # The database already exists and the task role has CREATE SCHEMA.
                # It owns only this task schema; no administrator password is needed.
                self._reset_delegated_schema()
            else:
                # Reuse the benchmark's destination reset exactly; no grader files enter the container.
                agents_path = self.spec.official_repo / "agents"
                sys.path.insert(0, str(agents_path))
                try:
                    from common import prepare_destination
                    prepare_destination("snowflake", self.spec.public_dir, self.spec.credential_path)
                finally:
                    sys.path.remove(str(agents_path))
                self._grant_database_usage()
            cache = self.root.parent / "terraform-plugin-cache"
            cache.mkdir(parents=True, exist_ok=True)
            mirror = os.environ.get("ELT_RLVR_TERRAFORM_MIRROR")
            cmd = [
                "docker", "run", "-d", "--rm", "--name", self.container,
                "--network", self.network,
                "-v", f"{self.root}:/workspace",
                "-v", f"{cache}:/tf-plugin-cache",
                "-e", "TF_PLUGIN_CACHE_DIR=/tf-plugin-cache",
                "-w", "/workspace", self.image, "sleep", "infinity",
            ]
            ca_bundle = os.environ.get("ELT_RLVR_CA_BUNDLE")
            if ca_bundle:
                ca_path = Path(ca_bundle).resolve(strict=True)
                if not ca_path.is_file():
                    raise ValueError("CA bundle must be a file")
                cmd[cmd.index("-w"):cmd.index("-w")] = [
                    "-v", f"{ca_path}:/rlvr-ca.pem:ro",
                    "-e", "REQUESTS_CA_BUNDLE=/rlvr-ca.pem",
                    "-e", "SSL_CERT_FILE=/rlvr-ca.pem",
                    "-e", "CURL_CA_BUNDLE=/rlvr-ca.pem",
                ]
            if mirror:
                mirror_path = Path(mirror).resolve(strict=True)
                if not mirror_path.is_dir():
                    raise ValueError("Terraform provider mirror must be a directory")
                cli_config = self.root / "terraform.rc"
                cli_config.write_text(
                    'provider_installation {\n'
                    '  filesystem_mirror {\n'
                    '    path = "/tf-mirror"\n'
                    '    include = ["registry.terraform.io/airbytehq/airbyte"]\n'
                    '  }\n'
                    '  direct {\n'
                    '    exclude = ["registry.terraform.io/airbytehq/airbyte"]\n'
                    '  }\n'
                    '}\n',
                    encoding="utf-8",
                )
                cmd[cmd.index("-w"):cmd.index("-w")] = [
                    "-v", f"{mirror_path}:/tf-mirror:ro",
                    "-e", "TF_CLI_CONFIG_FILE=/workspace/terraform.rc",
                ]
            self._command(cmd, timeout=60)
            self.started = True
        except BaseException:
            self.close()
            raise

    def _apply_el_credentials(self) -> None:
        """Overlay optional EL-only credentials onto this rollout's private copy."""
        user = os.environ.get("ELT_RLVR_SNOWFLAKE_EL_USER")
        password = os.environ.get("ELT_RLVR_SNOWFLAKE_EL_PASSWORD")
        if user is None and password is None:
            return
        if not user or not password:
            raise ValueError(
                "Set both ELT_RLVR_SNOWFLAKE_EL_USER and "
                "ELT_RLVR_SNOWFLAKE_EL_PASSWORD, or neither"
            )

        config_path = self.root / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        warehouse = config["snowflake"]["config"]
        warehouse["username"] = user
        warehouse["password"] = password
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        self._secrets.append(password)

    def _grant_database_usage(self) -> None:
        """Let the task's EL role access the database created by the reset helper."""
        import snowflake.connector

        assert self.spec.credential_path is not None
        admin = json.loads(self.spec.credential_path.read_text(encoding="utf-8"))
        cfg = yaml.safe_load((self.root / "config.yaml").read_text(encoding="utf-8"))["snowflake"]["config"]
        database = '"' + str(cfg["database"]).upper().replace('"', '""') + '"'
        role = '"' + str(cfg["role"]).upper().replace('"', '""') + '"'
        with snowflake.connector.connect(**admin) as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"GRANT USAGE ON DATABASE {database} TO ROLE {role}")

    def _reset_delegated_schema(self) -> None:
        """Reset only the task schema using the restricted EL role."""
        import snowflake.connector

        from .backend import valid_identifier

        assert self.spec.credential_path is not None
        credential = json.loads(self.spec.credential_path.read_text(encoding="utf-8"))
        cfg = yaml.safe_load((self.root / "config.yaml").read_text(encoding="utf-8"))["snowflake"]["config"]
        if credential.get("role", "").upper() != str(cfg["role"]).upper():
            raise ValueError("Delegated reset credential must use the task EL role")
        if credential.get("user", "").upper() != str(cfg.get("user") or cfg.get("username")).upper():
            raise ValueError("Delegated reset credential must use the task EL user")
        database = valid_identifier(str(cfg["database"])).upper()
        schema = valid_identifier(str(cfg["schema"])).upper()
        with snowflake.connector.connect(**credential) as connection:
            with connection.cursor() as cursor:
                cursor.execute(f'DROP SCHEMA IF EXISTS "{database}"."{schema}" CASCADE')
                cursor.execute(f'CREATE SCHEMA "{database}"."{schema}"')

    def _write_dbt_scaffold(self) -> None:
        cfg = yaml.safe_load((self.root / "config.yaml").read_text(encoding="utf-8"))
        warehouse = cfg["snowflake"]["config"]
        self._secrets = [str(value) for key, value in warehouse.items()
                         if key.lower() in {"password", "secret", "token", "access_key_id"} and value]
        self._secrets += [
            str(cfg["Airbyte"]["config"].get(key, ""))
            for key in ("password", "client_id", "client_secret")
        ]
        elt = self.root / "elt"
        (elt / "models").mkdir(parents=True, exist_ok=True)
        (elt / "dbt_project.yml").write_text(
            "name: elt_models\nversion: '1.0'\nconfig-version: 2\n"
            "profile: elt_models\nmodel-paths: ['models']\n"
            "models:\n  elt_models:\n    +materialized: table\n",
            encoding="utf-8",
        )
        profile = {"elt_models": {"target": "dev", "outputs": {"dev": {
            "type": "snowflake", "account": warehouse["account"],
            "user": warehouse.get("user") or warehouse.get("username"),
            "password": warehouse["password"],
            "role": warehouse.get("role"), "warehouse": warehouse["warehouse"],
            "database": warehouse["database"], "schema": warehouse["schema"],
            "threads": 2,
        }}}}
        (elt / "profiles.yml").write_text(yaml.safe_dump(profile), encoding="utf-8")

    def _command(self, argv: list[str], *, timeout: int) -> str:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        output = (result.stdout + "\n" + result.stderr)[-8000:]
        for secret in self._secrets:
            if secret:
                output = output.replace(secret, "[REDACTED]")
        if result.returncode:
            raise RuntimeError(f"Command failed ({result.returncode}): {output[-3000:]}")
        return output[-3000:]

    def write_terraform(self, filename: str, text: str) -> str:
        if not self.started:
            raise RuntimeError("Runtime is not started")
        parts = filename.replace("\\", "/").split("/")
        if len(parts) > 1:
            if parts[0] not in {"elt", "_elt"} or not all(
                re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", part) for part in parts[1:-1]
            ):
                raise ValueError("Terraform path must be inside elt/ and cannot traverse directories")
        basename = parts[-1]
        if basename == "main.tf" or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*\.tf", basename):
            raise ValueError("Use a new .tf filename such as trains.tf; main.tf is immutable")
        if len(text) > 100_000:
            raise ValueError("Terraform file is too large")
        lower = text.lower()
        for forbidden in ("provisioner", "local-exec", "remote-exec", "module ", 'output "', 'backend "'):
            if forbidden in lower:
                raise ValueError(f"Forbidden Terraform construct: {forbidden}")
        if re.search(r'\bdata\s+"', text, re.I):
            raise ValueError("Terraform data sources are not allowed")
        resources = re.findall(r'\bresource\s+"([^"]+)"', text, re.I)
        providers = re.findall(r'\bprovider\s+"([^"]+)"', text, re.I)
        if any(not kind.startswith("airbyte_") for kind in resources):
            raise ValueError("Only Airbyte Terraform resources are allowed")
        if any(kind != "airbyte" for kind in providers):
            raise ValueError("Only the Airbyte provider is allowed")
        if any(secret in text for secret in self._secrets if len(secret) > 3):
            raise ValueError("Do not embed credentials in Terraform; reference config.yaml with yamldecode")
        (self.root / "elt" / basename).write_text(text, encoding="utf-8")
        return f"wrote elt/{basename}"

    def configure_sources(self, tables: list[str]) -> str:
        """Render an Airbyte plan from the task's declared source metadata."""
        from .airbyte_plan import render_snowflake_plan

        run_name = "run_" + self.container.rsplit("-", 1)[-1]
        plan = render_snowflake_plan(self.spec, tables, run_name)
        self.write_terraform("pipeline.tf", plan)
        return f"configured Airbyte sources: {', '.join(tables)}"

    def run_extract_load(self) -> str:
        if not self.started:
            raise RuntimeError("Runtime is not started")
        files = [p for p in (self.root / "elt").glob("*.tf") if p.name != "main.tf"]
        if not files:
            raise ValueError("Configure Airbyte resources in a new .tf file first")
        init = self._command(
            ["docker", "exec", self.container, "terraform", "-chdir=/workspace/elt", "init", "-input=false"],
            timeout=300,
        )
        apply = self._command(
            ["docker", "exec", self.container, "terraform", "-chdir=/workspace/elt", "apply", "-auto-approve", "-input=false", "-parallelism=1"],
            timeout=900,
        )
        sync = self._command(
            ["docker", "exec", self.container, "python", "/workspace/rlvr_airbyte_sync.py"],
            timeout=1250,
        )
        return f"Terraform init: {init[-500:]}\nApply: {apply[-1200:]}\nSync: {sync[-1200:]}"

    def write_model(self, name: str, sql: str) -> str:
        if name not in self.spec.target_tables:
            raise ValueError("Only requested final models may be written")
        from .backend import select_only
        (self.root / "elt/models" / f"{name}.sql").write_text(select_only(sql), encoding="utf-8")
        return f"wrote elt/models/{name}.sql"

    def run_transforms(self) -> str:
        if not self.started:
            raise RuntimeError("Runtime is not started")
        return self._command([
            "docker", "exec", self.container, "dbt", "run",
            "--project-dir", "/workspace/elt", "--profiles-dir", "/workspace/elt",
        ], timeout=900)

    def close(self) -> None:
        try:
            if self.started:
                subprocess.run(["docker", "rm", "-f", self.container], capture_output=True, timeout=30)
        finally:
            self.started = False
            # Keep the submitted HCL and SQL for review, remove copied secrets/state.
            for file in (self.root / "config.yaml", self.root / "elt/profiles.yml",
                         self.root / "elt/terraform.tfstate", self.root / "elt/terraform.tfstate.backup"):
                file.unlink(missing_ok=True)
