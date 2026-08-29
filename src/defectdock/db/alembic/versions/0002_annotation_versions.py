"""Track immutable annotation versions and the active dataset head."""

from alembic import op

revision = "0002_annotation_versions"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE annotation_versions (
            annotation_version_id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
            source TEXT NOT NULL,
            format TEXT NOT NULL,
            root_dir TEXT NOT NULL,
            manifest_path TEXT NOT NULL,
            manifest_sha256 TEXT NOT NULL,
            labeled_count INTEGER NOT NULL CHECK(labeled_count >= 0),
            unlabeled_count INTEGER NOT NULL CHECK(unlabeled_count >= 0),
            created_at TEXT NOT NULL,
            UNIQUE(dataset_id, annotation_version_id),
            UNIQUE(dataset_id, manifest_sha256)
        )
        """
    )
    op.execute(
        "CREATE INDEX idx_annotation_versions_dataset "
        "ON annotation_versions(dataset_id, created_at DESC)"
    )
    op.execute(
        """
        CREATE TABLE dataset_annotation_heads (
            dataset_id TEXT PRIMARY KEY REFERENCES datasets(dataset_id) ON DELETE CASCADE,
            annotation_version_id TEXT NOT NULL UNIQUE,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(dataset_id, annotation_version_id)
                REFERENCES annotation_versions(dataset_id, annotation_version_id)
                ON DELETE CASCADE
        )
        """
    )


def downgrade() -> None:
    op.drop_table("dataset_annotation_heads")
    op.drop_index("idx_annotation_versions_dataset", table_name="annotation_versions")
    op.drop_table("annotation_versions")

