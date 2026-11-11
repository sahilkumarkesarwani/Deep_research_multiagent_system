"""
database.py — SQLite database layer using SQLAlchemy.

Tables
------
users          : user accounts (id, username, email, hashed_password, created_at)
research_runs  : one row per research run, linked to a user
chat_messages  : follow-up Q&A messages, linked to a research run
"""

import json
from datetime import datetime

from sqlalchemy import (
    Column, DateTime, ForeignKey, Integer, String, Text, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

DATABASE_URL = "sqlite:///research_agent.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, index=True)
    username   = Column(String(50),  unique=True, nullable=False, index=True)
    email      = Column(String(120), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    runs = relationship("ResearchRun", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User id={self.id} username={self.username!r}>"


class ResearchRun(Base):
    __tablename__ = "research_runs"

    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False)
    topic           = Column(String(300), nullable=False)
    search_results  = Column(Text, default="")
    scraped_content = Column(Text, default="")
    report          = Column(Text, default="")
    critique        = Column(Text, default="")
    # Timings, warnings, error stored as JSON strings
    timings_json    = Column(Text, default="{}")
    warnings_json   = Column(Text, default="{}")
    error_json      = Column(Text, default="null")
    created_at      = Column(DateTime, default=datetime.utcnow)
    finished_at     = Column(DateTime, nullable=True)

    user     = relationship("User", back_populates="runs")
    messages = relationship("ChatMessage", back_populates="run", cascade="all, delete-orphan",
                            order_by="ChatMessage.id")

    def __repr__(self):
        return f"<ResearchRun id={self.id} topic={self.topic!r}>"


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id         = Column(Integer, primary_key=True, index=True)
    run_id     = Column(Integer, ForeignKey("research_runs.id"), nullable=False)
    role       = Column(String(16), nullable=False)   # "user" | "assistant"
    content    = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("ResearchRun", back_populates="messages")

    def __repr__(self):
        return f"<ChatMessage id={self.id} role={self.role!r}>"

def init_db():
    """Create all tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)

def create_user(username: str, email: str, password_hash: str) -> User:
    """Insert a new user and return the ORM object. Raises on duplicate."""
    with SessionLocal() as db:
        user = User(username=username, email=email, password_hash=password_hash)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user


def get_user_by_username(username: str) -> User | None:
    with SessionLocal() as db:
        return db.query(User).filter(User.username == username).first()


def get_user_by_email(email: str) -> User | None:
    with SessionLocal() as db:
        return db.query(User).filter(User.email == email).first()


def get_user_by_id(user_id: int) -> User | None:
    with SessionLocal() as db:
        return db.query(User).filter(User.id == user_id).first()

def save_research_run(user_id: int, state: dict) -> int:
    """
    Persist (or update) a research run from the in-memory `state` dict
    used by app.py.  Returns the database row id.
    """
    with SessionLocal() as db:
        run_id = state.get("db_id")
        if run_id:
            run = db.query(ResearchRun).filter(ResearchRun.id == run_id).first()
        else:
            run = None

        # Convert integer-keyed dicts to string-keyed for JSON serialisation
        timings  = {str(k): v for k, v in state.get("timings", {}).items()}
        warnings = {str(k): v for k, v in state.get("warnings", {}).items()}

        if run is None:
            run = ResearchRun(user_id=user_id, topic=state.get("topic", ""))
            db.add(run)

        run.search_results  = state.get("search_results", "") or ""
        run.scraped_content = state.get("scraped_content", "") or ""
        run.report          = state.get("report", "") or ""
        run.critique        = state.get("critique", "") or ""
        run.timings_json    = json.dumps(timings)
        run.warnings_json   = json.dumps(warnings)
        run.error_json      = json.dumps(state.get("error"))
        if state.get("finished"):
            run.finished_at = datetime.utcnow()

        db.commit()
        db.refresh(run)

        # Sync chat messages — wipe existing and re-insert
        db.query(ChatMessage).filter(ChatMessage.run_id == run.id).delete()
        for msg in state.get("chat", []):
            db.add(ChatMessage(run_id=run.id, role=msg["role"], content=msg["content"]))
        db.commit()

        return run.id


def load_research_runs(user_id: int) -> list[dict]:
    """
    Return all research runs for a user as a list of state dicts
    compatible with app.py's `ss.history` format.
    """
    with SessionLocal() as db:
        runs = (
            db.query(ResearchRun)
            .filter(ResearchRun.user_id == user_id)
            .order_by(ResearchRun.created_at.asc())
            .all()
        )
        result = []
        for run in runs:
            # Restore integer-keyed dicts
            timings  = {int(k): v for k, v in json.loads(run.timings_json or "{}").items()}
            warnings = {int(k): v for k, v in json.loads(run.warnings_json or "{}").items()}
            error    = json.loads(run.error_json or "null")
            chat     = [
                {"role": m.role, "content": m.content}
                for m in db.query(ChatMessage)
                              .filter(ChatMessage.run_id == run.id)
                              .order_by(ChatMessage.id.asc())
                              .all()
            ]
            state = {
                "db_id"          : run.id,
                "topic"          : run.topic,
                "search_results" : run.search_results,
                "scraped_content": run.scraped_content,
                "report"         : run.report,
                "critique"       : run.critique,
                "timings"        : timings,
                "warnings"       : warnings,
                "error"          : error,
                "chat"           : chat,
                "finished"       : run.finished_at.strftime("%H:%M") if run.finished_at else None,
                "total_time"     : sum(timings.values()),
            }
            result.append(state)
        return result


def delete_research_run(run_id: int, user_id: int) -> bool:
    """Delete a run if it belongs to user_id. Returns True on success."""
    with SessionLocal() as db:
        run = db.query(ResearchRun).filter(
            ResearchRun.id == run_id,
            ResearchRun.user_id == user_id,
        ).first()
        if run is None:
            return False
        db.delete(run)
        db.commit()
        return True


def clear_all_runs(user_id: int):
    """Delete every research run (and cascaded chat messages) for a user."""
    with SessionLocal() as db:
        db.query(ResearchRun).filter(ResearchRun.user_id == user_id).delete()
        db.commit()


def add_chat_message(run_id: int, role: str, content: str) -> ChatMessage:
    """Append a single chat message to an existing research run."""
    with SessionLocal() as db:
        msg = ChatMessage(run_id=run_id, role=role, content=content)
        db.add(msg)
        db.commit()
        db.refresh(msg)
        return msg
