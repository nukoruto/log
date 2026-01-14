
import os
import sys
import torch
from models_lstm.utils import resolve_device

def test_resolve_device():
    print(f"Torch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Device 0 name: {torch.cuda.get_device_name(0)}")

    # Test 1: Default (Unset GPU_MODE)
    if "GPU_MODE" in os.environ:
        del os.environ["GPU_MODE"]
    
    device = resolve_device()
    print(f"[Test 1] GPU_MODE unset -> Resolved: {device}")
    
    if torch.cuda.is_available():
        assert device.type == "cuda", "Should default to CUDA if available"
    else:
        assert device.type == "cpu", "Should default to CPU if CUDA not available"

    # Test 2: Explicit CPU
    os.environ["GPU_MODE"] = "cpu"
    device = resolve_device()
    print(f"[Test 2] GPU_MODE='cpu' -> Resolved: {device}")
    assert device.type == "cpu"

    print("All tests passed!")

if __name__ == "__main__":
    try:
        test_resolve_device()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
