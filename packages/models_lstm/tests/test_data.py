import sys
from pathlib import Path
from typing import List

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from models_lstm.data import (  # noqa: E402
    OpCategoryEncoder,
    SequenceExample,
    build_sequence_examples,
    group_kfold_split,
    load_contract_dataframe,
    load_contract_sequences,
)


@pytest.fixture
def sample_contract_records() -> List[dict]:
    return [
        {
            "timestamp_utc": "2023-01-01T00:00:00Z",
            "uid": "u1",
            "session_id": "s1",
            "method": "GET",
            "path": "/a",
            "referer": "-",
            "user_agent": "ua",
            "ip": "1.1.1.1",
            "op_category": "cat_a",
        },
        {
            "timestamp_utc": "2023-01-01T00:00:30Z",
            "uid": "u1",
            "session_id": "s1",
            "method": "POST",
            "path": "/b",
            "referer": "-",
            "user_agent": "ua",
            "ip": "1.1.1.1",
            "op_category": "cat_b",
        },
        {
            "timestamp_utc": "2023-01-01T00:02:00Z",
            "uid": "u1",
            "session_id": "s1",
            "method": "GET",
            "path": "/c",
            "referer": "-",
            "user_agent": "ua",
            "ip": "1.1.1.1",
            "op_category": "cat_a",
        },
        {
            "timestamp_utc": "2023-01-02T00:00:00Z",
            "uid": "u2",
            "session_id": "s2",
            "method": "GET",
            "path": "/d",
            "referer": "-",
            "user_agent": "ua",
            "ip": "2.2.2.2",
            "op_category": "cat_c",
        },
        {
            "timestamp_utc": "2023-01-02T00:10:00Z",
            "uid": "u2",
            "session_id": "s2",
            "method": "POST",
            "path": "/e",
            "referer": "-",
            "user_agent": "ua",
            "ip": "2.2.2.2",
            "op_category": "cat_c",
        },
    ]


def test_load_contract_dataframe_missing_column(tmp_path):
    csv_path = tmp_path / "contract.csv"
    csv_path.write_text("timestamp_utc\n2023-01-01T00:00:00Z\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required columns"):  # noqa: PT011
        load_contract_dataframe(csv_path)


def test_load_contract_dataframe_rejects_reordered_or_extra_columns(tmp_path):
    csv_path = tmp_path / "contract.csv"
    header = [
        "uid",
        "timestamp_utc",
        "session_id",
        "method",
        "path",
        "referer",
        "user_agent",
        "ip",
        "op_category",
        "extra_field",
    ]
    row = [
        "u1",
        "2023-01-01T00:00:00Z",
        "s1",
        "GET",
        "/",
        "-",
        "ua",
        "1.1.1.1",
        "cat",
        "unexpected",
    ]
    csv_path.write_text(
        "\n".join([",".join(header), ",".join(row)]) + "\n", encoding="utf-8"
    )

    with pytest.raises(
        ValueError, match="Invalid contract CSV header"
    ) as excinfo:  # noqa: PT011
        load_contract_dataframe(csv_path)

    message = str(excinfo.value)
    assert "header order" in message
    assert "unexpected columns: extra_field" in message


def test_build_sequence_examples_produces_z_clipped(sample_contract_records):
    sequences, encoder, stats = build_sequence_examples(sample_contract_records)

    assert isinstance(encoder, OpCategoryEncoder)
    assert stats["median_delta_seconds"] == pytest.approx(30.0)
    assert stats["mad_delta_seconds"] == pytest.approx(30.0)
    assert encoder.vocab_size == 3

    first = next(seq for seq in sequences if seq.session_id == "s1")
    assert isinstance(first, SequenceExample)
    timestamp_pairs = zip(first.timestamps, first.timestamps[1:])
    assert all(later >= earlier for earlier, later in timestamp_pairs)
    assert all(-5.0 <= value <= 5.0 for value in first.z_clipped)

    decoded_ops = encoder.inverse_transform(first.op_indices)
    assert list(decoded_ops) == ["cat_a", "cat_b", "cat_a"]


def test_group_kfold_split_by_session(sample_contract_records):
    sequences, _, _ = build_sequence_examples(sample_contract_records)
    splits = list(group_kfold_split(sequences, n_splits=2, group_level="session"))
    assert len(splits) == 2

    for train_idx, test_idx in splits:
        train_sessions = {sequences[i].session_id for i in train_idx}
        test_sessions = {sequences[i].session_id for i in test_idx}
        assert train_sessions.isdisjoint(test_sessions)


def test_group_kfold_split_by_user(sample_contract_records):
    sequences, _, _ = build_sequence_examples(sample_contract_records)
    splits = list(group_kfold_split(sequences, n_splits=2, group_level="user"))

    for train_idx, test_idx in splits:
        train_users = {sequences[i].uid for i in train_idx}
        test_users = {sequences[i].uid for i in test_idx}
        assert train_users.isdisjoint(test_users)


@pytest.mark.parametrize("group_level", ["session", "user"])
def test_group_kfold_split_is_deterministic(sample_contract_records, group_level):
    sequences, _, _ = build_sequence_examples(sample_contract_records)

    first = list(group_kfold_split(sequences, n_splits=2, group_level=group_level))
    second = list(group_kfold_split(sequences, n_splits=2, group_level=group_level))

    assert first == second


def test_group_kfold_invalid_level(sample_contract_records):
    sequences, _, _ = build_sequence_examples(sample_contract_records)

    with pytest.raises(ValueError, match="Unsupported group_level"):  # noqa: PT011
        list(group_kfold_split(sequences, n_splits=2, group_level="invalid"))


def test_load_contract_sequences_roundtrip(sample_contract_records, tmp_path):
    csv_path = tmp_path / "contract.csv"
    columns = list(sample_contract_records[0].keys())
    rows = [",".join(columns)]
    for record in sample_contract_records:
        rows.append(",".join(str(record[column]) for column in columns))
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    sequences, encoder, stats = load_contract_sequences(csv_path)

    assert len(sequences) == 2
    assert encoder.vocab_size == 3
    assert stats["clip_value"] == 5.0


def test_build_sequence_examples_rejects_naive_timestamp(sample_contract_records):
    records = [dict(record) for record in sample_contract_records]
    records[0]["timestamp_utc"] = "2023-01-01T00:00:00"

    with pytest.raises(ValueError, match="UTC-normalized contract data"):  # noqa: PT011
        build_sequence_examples(records)


def test_build_sequence_examples_rejects_non_utc_offset(sample_contract_records):
    records = [dict(record) for record in sample_contract_records]
    records[0]["timestamp_utc"] = "2023-01-01T09:00:00+09:00"

    with pytest.raises(ValueError, match="UTC-normalized contract data"):  # noqa: PT011
        build_sequence_examples(records)
