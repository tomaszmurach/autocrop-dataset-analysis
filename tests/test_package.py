"""Smoke tests for the package scaffold."""

import unittest

import autocrop_analysis


class PackageImportTests(unittest.TestCase):
    def test_package_exposes_version(self) -> None:
        self.assertEqual(autocrop_analysis.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
