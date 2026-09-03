from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from extract_opera_ukcovid import activate_opera_namespace  # noqa: E402


class OperaNamespaceTest(unittest.TestCase):
    def test_existing_wrong_src_package_is_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrong = root / "wrong"
            opera = root / "OPERA"
            for package in (wrong / "src", opera / "src", opera / "src" / "model"):
                package.mkdir(parents=True, exist_ok=True)
                (package / "__init__.py").write_text("")
            (wrong / "src" / "marker.py").write_text("VALUE = 'wrong'\n")
            (opera / "src" / "model" / "models_cola.py").write_text("VALUE = 'opera'\n")
            (opera / "src" / "util.py").write_text("VALUE = 'opera-util'\n")
            sys.path.insert(0, str(wrong))
            try:
                import src.marker  # type: ignore  # noqa: F401
                activate_opera_namespace(opera)
                module = importlib.import_module("src.model.models_cola")
                self.assertEqual(module.VALUE, "opera")
                package = importlib.import_module("src")
                self.assertIn((opera / "src").resolve(),
                              [Path(item).resolve() for item in package.__path__])
            finally:
                for name in list(sys.modules):
                    if name == "src" or name.startswith("src."):
                        del sys.modules[name]
                sys.path[:] = [item for item in sys.path
                               if item not in (str(wrong), str(opera))]


if __name__ == "__main__":
    unittest.main()
