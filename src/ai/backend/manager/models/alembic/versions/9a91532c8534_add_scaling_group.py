"""add-scaling-group

Revision ID: 9a91532c8534
Revises: c401d78cc7b9
Create Date: 2019-07-25 22:32:25.974046

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql import text

import ai.backend.manager.models.base

# revision identifiers, used by Alembic.
revision = "9a91532c8534"
down_revision = "c401d78cc7b9"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "scaling_groups",
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("description", sa.String(length=512), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True
        ),
        sa.Column("driver", sa.String(length=64), nullable=False),
        sa.Column("driver_opts", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("scheduler", sa.String(length=64), nullable=False),
        sa.Column("scheduler_opts", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint("name", name=op.f("pk_scaling_groups")),
    )
    op.create_index(
        op.f("ix_scaling_groups_is_active"), "scaling_groups", ["is_active"], unique=False
    )
    op.create_table(
        "sgroups_for_domains",
        sa.Column("scaling_group", sa.String(length=64), nullable=False),
        sa.Column("domain", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(
            ["domain"],
            ["domains.name"],
            name=op.f("fk_sgroups_for_domains_domain_domains"),
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["scaling_group"],
            ["scaling_groups.name"],
            name=op.f("fk_sgroups_for_domains_scaling_group_scaling_groups"),
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("scaling_group", "domain", name="uq_sgroup_domain"),
    )
    op.create_index(
        op.f("ix_sgroups_for_domains_domain"), "sgroups_for_domains", ["domain"], unique=False
    )
    op.create_index(
        op.f("ix_sgroups_for_domains_scaling_group"),
        "sgroups_for_domains",
        ["scaling_group"],
        unique=False,
    )
    op.create_table(
        "sgroups_for_groups",
        sa.Column("scaling_group", sa.String(length=64), nullable=False),
        sa.Column("group", ai.backend.manager.models.base.GUID(), nullable=False),
        sa.ForeignKeyConstraint(
            ["group"],
            ["groups.id"],
            name=op.f("fk_sgroups_for_groups_group_groups"),
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["scaling_group"],
            ["scaling_groups.name"],
            name=op.f("fk_sgroups_for_groups_scaling_group_scaling_groups"),
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("scaling_group", "group", name="uq_sgroup_ugroup"),
    )
    op.create_index(
        op.f("ix_sgroups_for_groups_group"), "sgroups_for_groups", ["group"], unique=False
    )
    op.create_index(
        op.f("ix_sgroups_for_groups_scaling_group"),
        "sgroups_for_groups",
        ["scaling_group"],
        unique=False,
    )
    op.create_table(
        "sgroups_for_keypairs",
        sa.Column("scaling_group", sa.String(length=64), nullable=False),
        sa.Column("access_key", sa.String(length=20), nullable=False),
        sa.ForeignKeyConstraint(
            ["access_key"],
            ["keypairs.access_key"],
            name=op.f("fk_sgroups_for_keypairs_access_key_keypairs"),
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["scaling_group"],
            ["scaling_groups.name"],
            name=op.f("fk_sgroups_for_keypairs_scaling_group_scaling_groups"),
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("scaling_group", "access_key", name="uq_sgroup_akey"),
    )
    op.create_index(
        op.f("ix_sgroups_for_keypairs_access_key"),
        "sgroups_for_keypairs",
        ["access_key"],
        unique=False,
    )
    op.create_index(
        op.f("ix_sgroups_for_keypairs_scaling_group"),
        "sgroups_for_keypairs",
        ["scaling_group"],
        unique=False,
    )

    # create the default sgroup
    query = """
    INSERT INTO scaling_groups
    VALUES (
        'default',
        'The default agent scaling group',
        't',
        now(),
        'static',
        '{}'::jsonb,
        'fifo',
        '{}'::jsonb
    );
    """
    connection = op.get_bind()
    connection.execute(text(query))
    query = """
    INSERT INTO sgroups_for_domains
    VALUES ('default', 'default');
    """
    connection.execute(text(query))
    op.add_column(
        "agents",
        sa.Column("scaling_group", sa.String(length=64), server_default="default", nullable=False),
    )
    op.create_index(op.f("ix_agents_scaling_group"), "agents", ["scaling_group"], unique=False)
    op.create_foreign_key(
        op.f("fk_agents_scaling_group_scaling_groups"),
        "agents",
        "scaling_groups",
        ["scaling_group"],
        ["name"],
    )
    op.add_column(
        "kernels",
        sa.Column("scaling_group", sa.String(length=64), server_default="default", nullable=False),
    )
    op.create_index(op.f("ix_kernels_scaling_group"), "kernels", ["scaling_group"], unique=False)
    op.create_foreign_key(
        op.f("fk_kernels_scaling_group_scaling_groups"),
        "kernels",
        "scaling_groups",
        ["scaling_group"],
        ["name"],
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(
        op.f("fk_kernels_scaling_group_scaling_groups"), "kernels", type_="foreignkey"
    )
    op.drop_index(op.f("ix_kernels_scaling_group"), table_name="kernels")
    op.drop_column("kernels", "scaling_group")
    op.drop_constraint(op.f("fk_agents_scaling_group_scaling_groups"), "agents", type_="foreignkey")
    op.drop_index(op.f("ix_agents_scaling_group"), table_name="agents")
    op.drop_column("agents", "scaling_group")
    op.drop_index(op.f("ix_sgroups_for_keypairs_scaling_group"), table_name="sgroups_for_keypairs")
    op.drop_index(op.f("ix_sgroups_for_keypairs_access_key"), table_name="sgroups_for_keypairs")
    op.drop_table("sgroups_for_keypairs")
    op.drop_index(op.f("ix_sgroups_for_groups_scaling_group"), table_name="sgroups_for_groups")
    op.drop_index(op.f("ix_sgroups_for_groups_group"), table_name="sgroups_for_groups")
    op.drop_table("sgroups_for_groups")
    op.drop_index(op.f("ix_sgroups_for_domains_scaling_group"), table_name="sgroups_for_domains")
    op.drop_index(op.f("ix_sgroups_for_domains_domain"), table_name="sgroups_for_domains")
    op.drop_table("sgroups_for_domains")
    op.drop_index(op.f("ix_scaling_groups_is_active"), table_name="scaling_groups")
    op.drop_table("scaling_groups")
    # ### end Alembic commands ###
