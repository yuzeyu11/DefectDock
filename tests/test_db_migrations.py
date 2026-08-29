import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from defectdock.db import (
    DatasetStore,
    backup_database,
    current_revision,
    head_revision,
    restore_database,
    upgrade_database,
)

LEGACY_DATASETS_SCHEMA = """
CREATE TABLE datasets (
    dataset_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    scene TEXT NOT NULL,
    labels_json TEXT NOT NULL,
    status TEXT NOT NULL,
    root_dir TEXT NOT NULL,
    image_count INTEGER NOT NULL DEFAULT 0,
    total_bytes INTEGER NOT NULL DEFAULT 0,
    cvat_task_id INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


class DatabaseMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.db_path = self.root / "defectdock.db"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_new_database_reaches_packaged_head(self):
        result = upgrade_database(self.db_path)
        self.assertEqual(result.current_revision, head_revision(self.db_path))
        self.assertEqual(current_revision(self.db_path), "0002_annotation_versions")
        with closing(sqlite3.connect(self.db_path)) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        self.assertIn("annotation_versions", tables)
        self.assertIn("dataset_annotation_heads", tables)

    def test_legacy_database_is_backed_up_and_preserved_during_upgrade(self):
        self._create_legacy_dataset()
        result = upgrade_database(self.db_path)
        self.assertIsNotNone(result.backup_path)
        self.assertTrue(Path(result.backup_path).is_file())
        store = DatasetStore(self.db_path)
        self.assertEqual(store.get_dataset("ds-legacy").name, "Legacy")

    def test_backup_and_restore_round_trip(self):
        store = DatasetStore(self.db_path)
        store.create_dataset("Before", "board", ["pit"], self.root / "dataset", dataset_id="ds-1")
        backup = backup_database(self.db_path, self.root / "release-backup.sqlite3")
        store.delete_dataset("ds-1")
        restore_database(self.db_path, backup)
        self.assertEqual(DatasetStore(self.db_path).get_dataset("ds-1").name, "Before")

    def test_failed_upgrade_restores_legacy_database(self):
        self._create_legacy_dataset()

        def destructive_failure(config, revision):
            with closing(sqlite3.connect(self.db_path)) as connection:
                connection.execute("DELETE FROM datasets")
                connection.commit()
            raise RuntimeError("simulated migration failure")

        with patch("defectdock.db.migration.command.upgrade", side_effect=destructive_failure):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                upgrade_database(self.db_path)
        with closing(sqlite3.connect(self.db_path)) as connection:
            row = connection.execute(
                "SELECT name FROM datasets WHERE dataset_id = 'ds-legacy'"
            ).fetchone()
        self.assertEqual(row, ("Legacy",))

    def _create_legacy_dataset(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(LEGACY_DATASETS_SCHEMA)
            connection.execute(
                """
                INSERT INTO datasets (
                    dataset_id, name, scene, labels_json, status, root_dir,
                    image_count, total_bytes, created_at, updated_at
                ) VALUES ('ds-legacy', 'Legacy', 'board', '["pit"]', 'draft', ?, 0, 0, 'now', 'now')
                """,
                (str(self.root / "legacy"),),
            )
            connection.commit()


if __name__ == "__main__":
    unittest.main()
