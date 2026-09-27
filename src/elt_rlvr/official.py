"""Official Snowflake task runtime using ELT-Bench's own Airbyte/Terraform/dbt image."""

from __future__ import annotations

import shutil
import re
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
        for item in self.spec.public_dir.iterdir():
            if item.name.endswith("_credential.json"):
                continue
            dest = self.root / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        shutil.copy2(Path(__file__).with_name("airbyte_sync.py"), self.root / "rlvr_airbyte_sync.py")
        self._write_dbt_scaffold()
        # Reuse the benchmark's destination reset exactly; no grader files enter the container.
        agents_path = self.spec.official_repo / "agents"
        sys.path.insert(0, str(agents_path))
        try:
            from common import prepare_destination
            prepare_destination("snowflake", self.spec.public_dir, self.spec.credential_path)
        finally:
            sys.path.remove(str(agents_path))
        cache = self.root.parent / "terraform-plugin-cache"
        cache.mkdir(parents=True, exist_ok=True)
        cmd = [
            "docker", "run", "-d", "--rm", "--name", self.container,
            "--network", self.network,
            "-v", f"{self.root}:/workspace",
            "-v", f"{cache}:/tf-plugin-cache",
            "-e", "TF_PLUGIN_CACHE_DIR=/tf-plugin-cache",
            "-w", "/workspace", self.image, "sleep", "infinity",
        ]
        self._command(cmd, timeout=60)
        self.started = True

    def _write_dbt_scaffold(self) -> None:
        cfg = yaml.safe_load((self.root / "config.yaml").read_text(encoding="utf-8"))
        warehouse = cfg["snowflake"]["config"]
        self._secrets = [str(value) for key, value in warehouse.items()
                         if key.lower() in {"password", "secret", "token", "access_key_id"} and value]
        self._secrets += [str(cfg["Airbyte"]["config"].get("password", ""))]
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
        if filename == "main.tf" or not filename.endswith(".tf") or "/" in filename or "\\" in filename:
            raise ValueError("Write only a new elt/*.tf file; main.tf is immutable")
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
        (self.root / "elt" / filename).write_text(text, encoding="utf-8")
        return f"wrote elt/{filename}"

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
            ["docker", "exec", self.container, "terraform", "-chdir=/workspace/elt", "apply", "-auto-approve", "-input=false"],
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
        if self.started:
            subprocess.run(["docker", "rm", "-f", self.container], capture_output=True, timeout=30)
            self.started = False
        # Keep the submitted HCL and SQL for review, remove copied secrets/state.
        for file in (self.root / "config.yaml", self.root / "elt/profiles.yml",
                     self.root / "elt/terraform.tfstate", self.root / "elt/terraform.tfstate.backup"):
            file.unlink(missing_ok=True)
