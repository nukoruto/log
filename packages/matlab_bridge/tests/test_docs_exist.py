"""ドキュメントサンプルの存在と記述を確認するテスト。"""

from __future__ import annotations

from pathlib import Path


def test_metrics_example_document_exists() -> None:
    """IAE/ISE/ITAEなどの計算例が記されたMATLABスクリプトを検証する。"""

    repo_root = Path(__file__).resolve().parents[3]
    doc_path = repo_root / "docs" / "matlab_examples" / "metrics_example.m"

    assert doc_path.exists(), "docs/matlab_examples/metrics_example.m が存在しません。"

    content = doc_path.read_text(encoding="utf-8")

    for keyword in ["IAE", "ISE", "ITAE", "Tr", "Ts", "OS"]:
        assert (
            keyword in content
        ), f"サンプルスクリプトに指標 {keyword} の計算例が含まれていません。"

