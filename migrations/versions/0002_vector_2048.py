"""Update chunk vector dimension to 2048.

Revision ID: 0002_vector_2048
Revises: 0001_init
Create Date: 2026-06-19
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002_vector_2048"
down_revision: str | None = "0001_init"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DELETE FROM chunks")
    op.execute("ALTER TABLE chunks ALTER COLUMN content_vector TYPE vector(2048)")


def downgrade() -> None:
    op.execute("DELETE FROM chunks")
    op.execute("ALTER TABLE chunks ALTER COLUMN content_vector TYPE vector(1536)")
