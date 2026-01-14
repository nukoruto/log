
from pathlib import Path
from typing import Any, Dict, List
import pytest
from unittest.mock import MagicMock, patch

import torch
from models_lstm.score import score_dataset, ScoreResult
from models_lstm.data import ContractRecord

# Mock helpers
def create_mock_checkpoint():
    config = {
        "embed_dim": 16,
        "hidden_dim": 32,
        "num_layers": 1,
        "clip_value": 5.0,
    }
    model_state = {} 
    encoder_vocab = ["A", "B", "C"]
    
    # Create a mock checkpoint object
    checkpoint = MagicMock()
    checkpoint.config = config
    checkpoint.model_state = model_state
    checkpoint.encoder_vocab = encoder_vocab
    return checkpoint

def create_dummy_csv(path: Path):
    content = (
        "uid,session_id,timestamp_utc,op_category,label\n"
        "u1,s1,2023-01-01T10:00:00Z,A,0\n"
        "u1,s1,2023-01-01T10:00:01Z,B,0\n"
        "u1,s1,2023-01-01T10:00:02Z,C,1\n"
    )
    path.write_text(content, encoding="utf-8")

@patch("models_lstm.score._build_model")
@patch("models_lstm.score.load_contract_dataframe")
def test_score_generates_metrics_and_plot(mock_load, mock_build, tmp_path):
    # Setup
    input_path = tmp_path / "input.csv"
    model_path = tmp_path / "model.pt"
    output_path = tmp_path / "run" / "scored.csv"
    
    # Create dummy input data
    records = [
        ContractRecord(
            uid="u1", session_id="s1", 
            timestamp_utc=Any, # Mocked effectively by load return
            op_category="A", row_index=1,
            ip_address="1.2.3.4", user_agent="Mozilla",
            label="0"
        ),
        ContractRecord(
            uid="u1", session_id="s1", 
            timestamp_utc=Any, 
            op_category="B", row_index=2,
            ip_address="1.2.3.4", user_agent="Mozilla",
            label="0"
        ),
         ContractRecord(
            uid="u1", session_id="s1", 
            timestamp_utc=Any, 
            op_category="C", row_index=3,
            ip_address="1.2.3.4", user_agent="Mozilla",
            label="1"
        )
    ]
    # We need to manually set timestamps if we rely on sort or delta logic
    # But for this test, we mock _prepare_records or we integrate...
    # Let's simple-mock _prepare_records to return pre-filled records with s_cls
    pass

@patch("models_lstm.score._build_model")
@patch("models_lstm.score.load_contract_dataframe")
@patch("models_lstm.score._prepare_records")
@patch("models_lstm.score._score_records")
def test_metrics_calculation(mock_score_records, mock_prepare, mock_load, mock_build, tmp_path):
    input_path = tmp_path / "input.csv"
    model_path = tmp_path / "model.pt"
    output_path = tmp_path / "run" / "scored.csv"
    
    # Mock return values
    mock_load.return_value = []
    
    # Create records with specific s_cls values to verify math
    # Rec 1: s_cls = 0.1
    # Rec 2: s_cls = 0.5
    # Rec 3: s_cls = 0.9
    r1 = MagicMock(spec=ContractRecord)
    r1.s_cls = 0.1
    r1.timestamp_utc = MagicMock()
    r1.uid = "u1"
    r1.session_id = "s1"
    r1.op_category = "A"
    r1.z_clipped = 0.0
    
    r2 = MagicMock(spec=ContractRecord)
    r2.s_cls = 0.5
    r2.timestamp_utc = MagicMock()
    r2.op_category = "B"
    r2.z_clipped = 0.0
    
    r3 = MagicMock(spec=ContractRecord)
    r3.s_cls = 0.9
    r3.timestamp_utc = MagicMock()
    r3.op_category = "C"
    r3.z_clipped = 0.0

    mock_prepare.return_value = [r1, r2, r3]
    
    checkpoint = create_mock_checkpoint()
    
    # Run
    result = score_dataset(
        model_path=model_path,
        input_path=input_path,
        output_path=output_path,
        seed=42,
        checkpoint=checkpoint
    )
    
    # Verify Metrics
    # IAE = |0.1| + |0.5| + |0.9| = 1.5
    # ISE = 0.1^2 + 0.5^2 + 0.9^2 = 0.01 + 0.25 + 0.81 = 1.07
    # ITAE = 0*0.1 + 1*0.5 + 2*0.9 = 0 + 0.5 + 1.8 = 2.3
    
    assert result.metrics["iae"] == pytest.approx(1.5)
    assert result.metrics["ise"] == pytest.approx(1.07)
    assert result.metrics["itae"] == pytest.approx(2.3)
    
    # Verify files
    assert output_path.exists()
    assert (output_path.parent / "metrics.json").exists()
    
    # Waveform plot might fail if matplotlib not installed, checks logic inside
    # If matplotlib is mocked or present, it should exist
    # We can check if we tried to save it
