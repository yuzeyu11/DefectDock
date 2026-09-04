"""Register immutable training snapshots for lifecycle traceability."""

from alembic import op

revision = "0005_training_snapshots"
down_revision = "0004_annotation_review"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE training_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
            annotation_version_id TEXT NOT NULL,
            snapshot_sha256 TEXT NOT NULL,
            manifest_path TEXT NOT NULL,
            manifest_sha256 TEXT NOT NULL,
            data_yaml TEXT NOT NULL,
            image_count INTEGER NOT NULL CHECK(image_count >= 0),
            train_count INTEGER NOT NULL CHECK(train_count >= 0),
            val_count INTEGER NOT NULL CHECK(val_count >= 0),
            seed INTEGER NOT NULL CHECK(seed >= 0),
            val_ratio REAL NOT NULL CHECK(val_ratio >= 0.05 AND val_ratio <= 0.5),
            created_at TEXT NOT NULL,
            FOREIGN KEY(dataset_id, annotation_version_id)
                REFERENCES annotation_versions(dataset_id, annotation_version_id)
                ON DELETE RESTRICT,
            UNIQUE(dataset_id, snapshot_sha256)
        )
        """
    )
    op.execute(
        "CREATE INDEX idx_training_snapshots_dataset "
        "ON training_snapshots(dataset_id, created_at DESC)"
    )


def downgrade() -> None:
    op.drop_index("idx_training_snapshots_dataset", table_name="training_snapshots")
    op.drop_table("training_snapshots")
