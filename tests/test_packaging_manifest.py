import tomllib
import unittest
from pathlib import Path


class PackagingManifestTests(unittest.TestCase):
    def test_gui_imported_root_modules_are_packaged(self):
        root = Path(__file__).resolve().parents[1]
        with (root / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream)
        modules = set(project["tool"]["setuptools"]["py-modules"])
        self.assertIn("load_and_label_eis", modules)


if __name__ == "__main__":
    unittest.main()
