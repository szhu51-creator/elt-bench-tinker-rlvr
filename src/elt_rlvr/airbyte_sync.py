"""Runs inside the official ELT-Bench container after Terraform apply."""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

import yaml


def main() -> int:
    root = Path("/workspace")
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))["Airbyte"]["config"]
    state = json.loads((root / "elt/terraform.tfstate").read_text(encoding="utf-8"))
    ids = sorted({
        inst["attributes"]["connection_id"]
        for resource in state.get("resources", []) if resource.get("type") == "airbyte_connection"
        for inst in resource.get("instances", [])
        if inst.get("attributes", {}).get("connection_id")
    })
    if not ids:
        print("No Airbyte connection IDs in Terraform state", file=sys.stderr)
        return 2
    base = str(config["server_url"]).rstrip("/")
    auth = base64.b64encode(f"{config['username']}:{config['password']}".encode()).decode()

    def request(method: str, path: str, payload: dict | None = None) -> dict:
        body = json.dumps(payload).encode() if payload is not None else None
        req = Request(base + path, data=body, method=method, headers={
            "Authorization": f"Basic {auth}", "accept": "application/json",
            "content-type": "application/json",
        })
        with urlopen(req, timeout=30) as response:
            return json.load(response)

    jobs = {}
    for connection_id in ids:
        reply = request("POST", "/jobs", {"jobType": "sync", "connectionId": connection_id})
        jobs[str(reply["jobId"])] = connection_id
    deadline = time.monotonic() + 1200
    done = {}
    while len(done) < len(jobs) and time.monotonic() < deadline:
        for job_id, connection_id in jobs.items():
            if job_id in done:
                continue
            status = str(request("GET", f"/jobs/{job_id}").get("status", "unknown")).lower()
            if status in {"succeeded", "failed", "cancelled", "canceled", "incomplete"}:
                done[job_id] = status
                print(json.dumps({"connection_id": connection_id, "job_id": job_id, "status": status}))
        if len(done) < len(jobs):
            time.sleep(10)
    if len(done) != len(jobs):
        print("Airbyte sync timeout", file=sys.stderr)
        return 3
    return 0 if all(s == "succeeded" for s in done.values()) else 4


if __name__ == "__main__":
    raise SystemExit(main())
