"""add_dotfiles_to_domains_and_groups

Revision ID: 25e903510fa1
Revises: 0d553d59f369
Create Date: 2020-09-11 17:00:00.564219

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "25e903510fa1"
down_revision = "0d553d59f369"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column(
        "domains",
        sa.Column("dotfiles", sa.LargeBinary(length=65536), nullable=False, server_default="\\x90"),
    )
    op.add_column(
        "groups",
        sa.Column("dotfiles", sa.LargeBinary(length=65536), nullable=True, server_default="\\x90"),
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("groups", "dotfiles")
    op.drop_column("domains", "dotfiles")
    # ### end Alembic commands ###
