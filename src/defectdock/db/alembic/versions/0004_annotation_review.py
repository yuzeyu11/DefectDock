"""Require explicit review for model-generated annotation versions."""

from alembic import op

revision = "0004_annotation_review"
down_revision = "0003_model_registry"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE annotation_versions ADD COLUMN "
        "review_status TEXT NOT NULL DEFAULT 'approved' "
        "CHECK(review_status IN ('candidate', 'approved'))"
    )
    op.execute("ALTER TABLE annotation_versions ADD COLUMN reviewed_by TEXT")
    op.execute("ALTER TABLE annotation_versions ADD COLUMN reviewed_at TEXT")


def downgrade() -> None:
    with op.batch_alter_table("annotation_versions") as batch:
        batch.drop_column("reviewed_at")
        batch.drop_column("reviewed_by")
        batch.drop_column("review_status")
