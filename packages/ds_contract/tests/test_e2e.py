"""E2E smoke test for ds-contract quickstart pipeline."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _script_path() -> Path:
    return _repo_root() / "scripts" / "quickstart.sh"


def _prepare_env() -> dict[str, str]:
    env = os.environ.copy()
    root = _repo_root()
    src_path = root / "packages" / "ds_contract" / "src"
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{src_path}" if not existing else f"{src_path}{os.pathsep}{existing}"
    )
    env["QUICKSTART_SEED"] = "202401"
    return env


def test_quickstart_pipeline_is_reproducible(tmp_path: Path) -> None:
    script = _script_path()
    assert script.exists(), "quickstart fixture script must exist"

    env = _prepare_env()

    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"

    subprocess.run(["bash", str(script), str(run1)], check=True, env=env)
    subprocess.run(["bash", str(script), str(run2)], check=True, env=env)

    contract1 = (run1 / "contract.csv").read_bytes()
    contract2 = (run2 / "contract.csv").read_bytes()
    assert contract1 == contract2

    session1 = (run1 / "sessioned.csv").read_bytes()
    session2 = (run2 / "sessioned.csv").read_bytes()
    assert session1 == session2

    delta1 = (run1 / "deltified.csv").read_bytes()
    delta2 = (run2 / "deltified.csv").read_bytes()
    assert delta1 == delta2

    meta_session = json.loads((run1 / "meta_session.json").read_text(encoding="utf-8"))
    meta_session_again = json.loads(
        (run2 / "meta_session.json").read_text(encoding="utf-8")
    )
    assert meta_session == meta_session_again

    meta_dt = json.loads((run1 / "meta_dt.json").read_text(encoding="utf-8"))
    meta_dt_again = json.loads((run2 / "meta_dt.json").read_text(encoding="utf-8"))
    assert meta_dt == meta_dt_again

    header = contract1.splitlines()[0]
    assert (
        header
        == b"timestamp_utc,uid,session_id,method,path,referer,user_agent,ip,op_category"
    )
