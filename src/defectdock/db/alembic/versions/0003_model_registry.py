"""Register immutable model artifacts and audited activation history."""

from alembic import op

revision = "0003_model_registry"
down_revision = "0002_annotation_versions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE model_versions (
            model_version_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL UNIQUE REFERENCES runs(run_id) ON DELETE RESTRICT,
            project TEXT NOT NULL,
            task TEXT NOT NULL,
            engine TEXT NOT NULL,
            architecture TEXT NOT NULL,
            dataset TEXT NOT NULL,
            dataset_version TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            artifact_path TEXT NOT NULL,
            artifact_sha256 TEXT NOT NULL,
            artifact_size INTEGER NOT NULL CHECK(artifact_size >= 0),
            run_manifest_path TEXT,
            run_manifest_sha256 TEXT,
            metrics_json TEXT,
            created_by TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    op.execute(
        "CREATE INDEX idx_model_versions_created ON model_versions(created_at DESC)"
    )
    op.execute(
        "CREATE INDEX idx_model_versions_project ON model_versions(project, created_at DESC)"
    )
    op.execute(
        """
        CREATE TABLE model_activation_head (
            slot TEXT PRIMARY KEY CHECK(slot = 'default'),
            model_version_id TEXT NOT NULL REFERENCES model_versions(model_version_id) ON DELETE RESTRICT,
            actor TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE TABLE model_activation_events (
            activation_id TEXT PRIMARY KEY,
            action TEXT NOT NULL CHECK(action IN ('activate', 'rollback')),
            model_version_id TEXT NOT NULL REFERENCES model_versions(model_version_id) ON DELETE RESTRICT,
            previous_model_version_id TEXT REFERENCES model_versions(model_version_id) ON DELETE RESTRICT,
            actor TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    op.execute(
        "CREATE INDEX idx_model_activation_events_created "
        "ON model_activation_events(created_at DESC, activation_id DESC)"
    )


def downgrade() -> None:
    op.drop_index("idx_model_activation_events_created", table_name="model_activation_events")
    op.drop_table("model_activation_events")
    op.drop_table("model_activation_head")
    op.drop_index("idx_model_versions_project", table_name="model_versions")
    op.drop_index("idx_model_versions_created", table_name="model_versions")
    op.drop_table("model_versions")
