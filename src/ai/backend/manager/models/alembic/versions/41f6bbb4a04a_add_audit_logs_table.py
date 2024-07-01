"""add audit_logs table

Revision ID: 41f6bbb4a04a
Revises: fdb2dcdb8811
Create Date: 2024-03-31 14:57:34.598304

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from ai.backend.manager.models.base import GUID

# revision identifiers, used by Alembic.
revision = "41f6bbb4a04a"
down_revision = "fdb2dcdb8811"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "audit_logs",
        sa.Column(
            "id",
            GUID,
            server_default=sa.text("uuid_generate_v4()"),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("user_id", GUID, nullable=False),
        sa.Column("access_key", sa.String(length=20), nullable=False),
        sa.Column("email", sa.String(length=64), nullable=False),
        sa.Column("action", sa.VARCHAR, nullable=False),
        sa.Column("data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "target_type",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column("target", sa.String(length=64), nullable=True),
        sa.Column("error", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("success", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("rest_resource", sa.String(length=256), nullable=True),
        sa.Column("gql_query", sa.String(length=1024), nullable=True),
    )


def downgrade():
    op.drop_table("audit_logs")
