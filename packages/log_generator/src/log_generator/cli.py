"""Command-line interface for the log generator."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

from .generate import (
    ScenarioSpecError,
    generate_anom_records,
    generate_normal_records,
    load_spec,
    write_audit_log,
    write_contract_csv,
    write_run_meta,
)


def _json_log(event: str, **payload: Any) -> None:
    message = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "component": "log-generator",
        "event": event,
        **payload,
    }
    click.echo(json.dumps(message))


@click.group()
def cli() -> None:
    """Entry point for the log-generator tool."""


@cli.command()
@click.option(
    "--spec", "spec_path", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option("--seed", type=int, required=True)
@click.option("--normal", "normal_path", type=click.Path(path_type=Path), required=True)
@click.option("--anom", "anom_path", type=click.Path(path_type=Path), required=True)
@click.option("--audit", "audit_path", type=click.Path(path_type=Path), required=True)
@click.option("--meta", "meta_path", type=click.Path(path_type=Path), required=True)
def run(
    spec_path: Path,
    seed: int,
    normal_path: Path,
    anom_path: Path,
    audit_path: Path,
    meta_path: Path,
) -> None:
    """Generate normal logs from the given scenario specification."""

    spec_sha = _sha256_file(spec_path)
    _json_log(
        "start", command="run", seed=seed, spec=str(spec_path), spec_sha256=spec_sha
    )

    try:
        spec = load_spec(spec_path)
        normal_records = generate_normal_records(spec, seed)
        anom_records, audit_entries = generate_anom_records(spec, seed)
        write_contract_csv(normal_path, normal_records)
        write_contract_csv(anom_path, anom_records)
        write_audit_log(audit_path, audit_entries)
        write_run_meta(meta_path, spec, seed, spec_sha)
    except ScenarioSpecError as exc:
        _json_log(
            "error",
            command="run",
            seed=seed,
            spec=str(spec_path),
            spec_sha256=spec_sha,
            normal=str(normal_path),
            anom=str(anom_path),
            audit=str(audit_path),
            meta=str(meta_path),
            message=str(exc),
        )
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive safeguard
        _json_log(
            "error",
            command="run",
            seed=seed,
            spec=str(spec_path),
            spec_sha256=spec_sha,
            normal=str(normal_path),
            anom=str(anom_path),
            audit=str(audit_path),
            meta=str(meta_path),
            message=str(exc),
        )
        raise

    _json_log(
        "complete",
        command="run",
        seed=seed,
        spec=str(spec_path),
        spec_sha256=spec_sha,
        normal=str(normal_path),
        anom=str(anom_path),
        audit=str(audit_path),
        meta=str(meta_path),
        normal_rows=len(normal_records),
        anom_rows=len(anom_records),
        audit_entries=len(audit_entries),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()
