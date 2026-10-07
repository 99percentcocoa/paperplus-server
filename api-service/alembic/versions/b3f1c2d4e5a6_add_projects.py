"""add projects; schools/students/scans belong to a project

Existing schools, students and scans are all PaperPlus. schools/students keep a DB-level
'paperplus' default (like the other defaults added in 7e9ecd69fc7a) so raw SQL inserts still work.

Revision ID: b3f1c2d4e5a6
Revises: 35c162409d67
Create Date: 2026-09-28 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b3f1c2d4e5a6'
down_revision = '35c162409d67'
branch_labels = None
depends_on = None


def upgrade() -> None:
    projects = op.create_table(
        'projects',
        sa.Column('project_code', sa.String(), nullable=False),
        sa.Column('project_name', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('project_code'),
    )
    op.bulk_insert(projects, [
        {'project_code': 'paperplus', 'project_name': 'PaperPlus'},
        {'project_code': 'navodaya', 'project_name': 'Navodaya'},
    ])

    for table in ('schools', 'students'):
        op.add_column(table, sa.Column('project_code', sa.String(), server_default='paperplus', nullable=False))
        op.create_index(op.f(f'ix_{table}_project_code'), table, ['project_code'])
        op.create_foreign_key(f'{table}_project_code_fkey', table, 'projects', ['project_code'], ['project_code'])

    op.add_column('scans', sa.Column('project_code', sa.String(), nullable=True))
    op.execute("UPDATE scans SET project_code = 'paperplus'")
    op.create_index(op.f('ix_scans_project_code'), 'scans', ['project_code'])
    op.create_foreign_key('scans_project_code_fkey', 'scans', 'projects', ['project_code'], ['project_code'])


def downgrade() -> None:
    for table in ('scans', 'students', 'schools'):
        op.drop_constraint(f'{table}_project_code_fkey', table, type_='foreignkey')
        op.drop_index(op.f(f'ix_{table}_project_code'), table_name=table)
        op.drop_column(table, 'project_code')
    op.drop_table('projects')
