"""add users and prediction ownership"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("users"):
        op.create_table(
            "users",
            sa.Column("id", sa.BigInteger(), sa.Identity(always=True), primary_key=True),
            sa.Column("email", sa.String(320), nullable=False),
            sa.Column("password_hash", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.UniqueConstraint("email", name="uq_users_email"),
        )
        op.create_index("ix_users_email", "users", ["email"], unique=True)
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("predictions")}
    if "user_id" not in columns:
        op.add_column("predictions", sa.Column("user_id", sa.BigInteger(), nullable=True))
        op.create_foreign_key("fk_predictions_user", "predictions", "users", ["user_id"], ["id"], ondelete="CASCADE")
        op.create_index("ix_predictions_user_id", "predictions", ["user_id"])


def downgrade():
    op.drop_column("predictions", "user_id")
    op.drop_table("users")
