"""add audit_logs table

Revision ID: 41f6bbb4a04a
Revises: 75ea2b136830
Create Date: 2024-03-31 14:57:34.598304

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from ai.backend.manager.models.base import GUID

# revision identifiers, used by Alembic.
revision = "41f6bbb4a04a"
down_revision = "75ea2b136830"
branch_labels = None
depends_on = None


auditlogaction_choices = (
    "CREATE",
    "CHANGE",
    "DELETE",
    "PURGE",
    "RESTORE",
)

auditlogaction = postgresql.ENUM(*auditlogaction_choices, name="auditlogaction")


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "audit_logs",
        sa.Column("user_id", GUID, nullable=False),
        sa.Column("access_key", sa.String(length=20), nullable=False),
        sa.Column("email", sa.String(length=64), nullable=False),
        sa.Column(
            "action", sa.Enum(*auditlogaction_choices, name="auditlogaction"), nullable=False
        ),
        sa.Column("data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "target_type",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column("target", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True
        ),
        sa.Column("success", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("rest_resource", sa.String(length=256), nullable=True),
        sa.Column("gql_query", sa.String(length=1024), nullable=True),
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table("audit_logs")
    auditlogaction.drop(op.get_bind())
    # ### end Alembic commands ###
