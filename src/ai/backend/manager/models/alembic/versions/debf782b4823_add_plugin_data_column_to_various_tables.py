"""add plugin_data column to various tables

Revision ID: debf782b4823
Revises: fdb2dcdb8811
Create Date: 2024-06-10 07:03:55.203832

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "debf782b4823"
down_revision = "fdb2dcdb8811"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column(
        "endpoints",
        sa.Column(
            "plugin_data",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
    )
    op.add_column(
        "sessions",
        sa.Column(
            "plugin_data",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
    )
    op.add_column(
        "users",
        sa.Column(
            "plugin_data",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
    )
    op.add_column(
        "vfolders",
        sa.Column(
            "plugin_data",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("vfolders", "plugin_data")
    op.drop_column("users", "plugin_data")
    op.drop_column("sessions", "plugin_data")
    op.drop_column("endpoints", "plugin_data")
    # ### end Alembic commands ###
