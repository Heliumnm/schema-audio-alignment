import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXTRACT = ROOT / "external_validation" / "cambridge_covid_sounds" / "src" / "extract_embeddings.py"
sys.path.insert(0, str(EXTRACT.parent))
SPEC = importlib.util.spec_from_file_location("shared_extract_embeddings", EXTRACT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_hear_short_recording_is_right_padded_without_normalisation():
    waveform = np.linspace(-0.25, 0.5, 16_000, dtype=np.float32)
    windows = MODULE.fixed_windows(waveform, MODULE.HEAR_WINDOW_SAMPLES)
    assert windows.shape == (1, 32_000)
    np.testing.assert_array_equal(windows[0, :16_000], waveform)
    np.testing.assert_array_equal(windows[0, 16_000:], np.zeros(16_000, dtype=np.float32))


def test_hear_long_recording_uses_ceil_full_coverage_windows():
    waveform = np.arange(72_000, dtype=np.float32)
    starts = MODULE.window_starts(len(waveform), MODULE.HEAR_WINDOW_SAMPLES)
    windows = MODULE.fixed_windows(waveform, MODULE.HEAR_WINDOW_SAMPLES)
    assert starts == [0, 20_000, 40_000]
    assert windows.shape == (3, 32_000)
    assert windows[0, 0] == 0
    assert windows[-1, -1] == 71_999


def test_projector_accepts_hear_512_dimensional_input():
    import pytest
    torch = pytest.importorskip("torch")
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from projector import ContrastiveProjectionHead

    output = ContrastiveProjectionHead(in_dim=512)(torch.zeros(3, 512))
    assert tuple(output.shape) == (3, 2560)
    assert torch.isfinite(output).all()
