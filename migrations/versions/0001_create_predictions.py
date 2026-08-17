"""create predictions table"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    if sa.inspect(bind).has_table("predictions"):
        return
    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), primary_key=True),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("image_name", sa.Text(), nullable=False),
        sa.Column("image_path", sa.Text()),
        sa.Column("image_hash", sa.CHAR(64)),
        sa.Column("predicted_class", sa.Text(), nullable=False),
        sa.Column("confidence", sa.REAL(), nullable=False),
        sa.Column("top_k_predictions", postgresql.JSONB()),
        sa.Column("inference_ms", sa.Numeric(10, 3), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_confidence"),
        sa.CheckConstraint("inference_ms >= 0", name="ck_inference_ms"),
    )
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"])
    op.create_index("ix_predictions_class", "predictions", ["predicted_class"])


def downgrade():
    op.drop_table("predictions")
