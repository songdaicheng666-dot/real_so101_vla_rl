import unittest

import real_so101_vla_rl


class PackageImportTest(unittest.TestCase):
    def test_package_version_is_exposed(self) -> None:
        self.assertEqual(real_so101_vla_rl.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()

