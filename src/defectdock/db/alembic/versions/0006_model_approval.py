"""Require approval before a registered model can be activated."""

from alembic import op

revision = "0006_model_approval"
down_revision = "0005_training_snapshots"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE model_versions ADD COLUMN "
        "approval_status TEXT NOT NULL DEFAULT 'candidate' "
        "CHECK(approval_status IN ('candidate', 'approved'))"
    )
    op.execute("ALTER TABLE model_versions ADD COLUMN approved_by TEXT")
    op.execute("ALTER TABLE model_versions ADD COLUMN approved_at TEXT")
    # Models activated before this migration already crossed the product's
    # explicit activation boundary. Preserve that decision and its latest actor.
    op.execute(
        """
        UPDATE model_versions
           SET approval_status = 'approved',
               approved_by = COALESCE(
                   (SELECT mae.actor FROM model_activation_events mae
                     WHERE mae.model_version_id = model_versions.model_version_id
                     ORDER BY mae.rowid DESC LIMIT 1),
                   created_by
               ),
               approved_at = COALESCE(
                   (SELECT mae.created_at FROM model_activation_events mae
                     WHERE mae.model_version_id = model_versions.model_version_id
                     ORDER BY mae.rowid DESC LIMIT 1),
                   created_at
               )
         WHERE EXISTS (
             SELECT 1 FROM model_activation_events mae
              WHERE mae.model_version_id = model_versions.model_version_id
         )
        """
    )


def downgrade() -> None:
    with op.batch_alter_table("model_versions") as batch:
        batch.drop_column("approved_at")
        batch.drop_column("approved_by")
        batch.drop_column("approval_status")
