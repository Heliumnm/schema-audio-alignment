import importlib.util
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parent / "src" / "extract_hear_ukcovid.py"
SPEC = importlib.util.spec_from_file_location("extract_hear_ukcovid", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_short_window_is_right_padded_and_unchanged():
    x = np.linspace(-0.3, 0.7, 16_000, dtype=np.float32)
    windows = MODULE.fixed_windows(x)
    assert windows.shape == (1, 32_000)
    np.testing.assert_array_equal(windows[0, :16_000], x)
    assert np.all(windows[0, 16_000:] == 0)


def test_long_window_count_and_endpoints_are_frozen():
    x = np.arange(72_000, dtype=np.float32)
    assert MODULE.window_starts(len(x)) == [0, 20_000, 40_000]
    windows = MODULE.fixed_windows(x)
    assert windows.shape == (3, 32_000)
    assert windows[0, 0] == 0 and windows[-1, -1] == 71_999


def test_window_self_test_runs_without_tensorflow():
    MODULE.self_test()

