from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.config import settings

# pool_recycle matters for Supabase: the pooler drops idle connections
# server-side, and without it SQLAlchemy hands you a dead socket.
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=2,
    pool_recycle=300,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db():
    """FastAPI dependency - one session per request, always closed."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()