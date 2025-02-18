from sqlalchemy.orm import Mapped, mapped_column

from .engine import Base

from uuid import UUID, uuid4
from datetime import datetime


class Analysis(Base):
    __tablename__ = "Analysis"

    text: Mapped[str] = mapped_column(nullable=False)
    result: Mapped[str] = mapped_column(nullable=False)
    accuracy: Mapped[int] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow())
    id: Mapped[UUID] = mapped_column(primary_key=True, default_factory=uuid4)
