"""Establish the legacy dataset and run metadata baseline."""

from alembic import op

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS datasets (
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
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_datasets_created_at ON datasets(created_at DESC)")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_images (
            image_id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
            original_name TEXT NOT NULL,
            stored_name TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(dataset_id, sha256),
            UNIQUE(dataset_id, stored_name)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_dataset_images_dataset "
        "ON dataset_images(dataset_id, created_at)"
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            project TEXT NOT NULL,
            task TEXT NOT NULL,
            engine TEXT NOT NULL,
            model TEXT NOT NULL,
            dataset TEXT NOT NULL,
            dataset_version TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            config_json TEXT NOT NULL,
            status TEXT NOT NULL,
            output_dir TEXT NOT NULL,
            metrics_json TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project, created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_runs_config_hash ON runs(config_hash)")


def downgrade() -> None:
    op.drop_table("runs")
    op.drop_table("dataset_images")
    op.drop_table("datasets")

