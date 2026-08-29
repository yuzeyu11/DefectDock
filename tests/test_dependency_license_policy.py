import unittest

from scripts.check_dependency_licenses import check


class DependencyLicensePolicyTests(unittest.TestCase):
    def test_accepts_licensecheck_and_pnpm_reports(self):
        self.assertEqual(check([{"name": "fastapi", "license": "MIT"}]), [])
        self.assertEqual(check({"MIT": [{"name": "react", "license": "MIT"}]}), [])

    def test_rejects_excluded_package_and_agpl(self):
        violations = check(
            [
                {"name": "ultralytics", "license": "AGPL-3.0"},
                {"name": "example", "license": "GNU Affero General Public License v3"},
            ]
        )
        self.assertTrue(any("forbidden package" in item for item in violations))
        self.assertEqual(sum("forbidden license" in item for item in violations), 2)


if __name__ == "__main__":
    unittest.main()
