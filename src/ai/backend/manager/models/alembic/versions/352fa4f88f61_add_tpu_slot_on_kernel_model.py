"""add tpu slot on kernel model

Revision ID: 352fa4f88f61
Revises: 57b523dec0e8
Create Date: 2018-11-12 11:39:30.613081

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "352fa4f88f61"
down_revision = "57b523dec0e8"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column("kernels", sa.Column("tpu_set", sa.ARRAY(sa.Integer()), nullable=True))
    op.add_column("kernels", sa.Column("tpu_slot", sa.Float(), nullable=False, server_default="0"))
    op.alter_column("kernels", "tpu_slot", server_default=None)
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("kernels", "tpu_slot")
    op.drop_column("kernels", "tpu_set")
    # ### end Alembic commands ###
