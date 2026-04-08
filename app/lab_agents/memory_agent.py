"""
Memory Agent (Lab 5)
=====================
A conversational agent with thread-based memory using LangGraph's
MemorySaver checkpointer. Includes note-taking tools that persist to CSV.
"""

import csv
import os
from pathlib import Path

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from app.config import get_llm


# ---------------------------------------------------------------------------
# CSV-backed notes store
# ---------------------------------------------------------------------------

NOTES_CSV_PATH = Path(__file__).resolve().parent.parent.parent / "notes.csv"


class NotesStore:
    """A simple key-value store backed by a CSV file."""

    def __init__(self, csv_path: Path = NOTES_CSV_PATH):
        self.csv_path = csv_path
        self.notes: dict[str, str] = {}
        self._load()

    def _load(self):
        """Load notes from CSV on startup."""
        if not self.csv_path.exists():
            return
        try:
            with open(self.csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.notes[row["key"]] = row["value"]
        except Exception:
            pass  # Start fresh if CSV is corrupted

    def _save(self):
        """Persist all notes to CSV."""
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["key", "value"])
            writer.writeheader()
            for key, value in self.notes.items():
                writer.writerow({"key": key, "value": value})

    def set(self, key: str, value: str):
        self.notes[key] = value
        self._save()

    def get(self, key: str) -> str | None:
        return self.notes.get(key)

    def delete(self, key: str) -> bool:
        if key in self.notes:
            del self.notes[key]
            self._save()
            return True
        return False

    def all(self) -> dict[str, str]:
        return dict(self.notes)


# Singleton store — loaded once at import time
notes_store = NotesStore()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Use Python syntax: '2+2', '10**3', 'sum([1,2,3])'."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"


@tool
def save_note(key: str, value: str) -> str:
    """Save a note with a key for later retrieval. Notes persist across server restarts."""
    notes_store.set(key, value)
    return f"Saved note '{key}': {value}"


@tool
def get_note(key: str) -> str:
    """Retrieve a previously saved note by its key."""
    value = notes_store.get(key)
    if value:
        return f"Note '{key}': {value}"
    available = list(notes_store.all().keys())
    return f"No note found with key '{key}'. Available keys: {available}"


@tool
def list_notes() -> str:
    """List all saved notes."""
    all_notes = notes_store.all()
    if not all_notes:
        return "No notes saved yet."
    lines = [f"  • {k}: {v}" for k, v in all_notes.items()]
    return "Saved notes:\n" + "\n".join(lines)


@tool
def delete_note(key: str) -> str:
    """Delete a saved note by its key."""
    if notes_store.delete(key):
        return f"Deleted note '{key}'."
    available = list(notes_store.all().keys())
    return f"No note found with key '{key}'. Available keys: {available}"


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

MEMORY_AGENT_SYSTEM = """You are a friendly conversational assistant.
You remember everything the user has told you in this session.
Refer back to earlier parts of the conversation when relevant.
You can save, retrieve, list, and delete notes — these persist across sessions."""


def build_memory_agent():
    """Create a ReAct agent backed by an in-memory checkpointer with CSV-persisted note tools."""
    llm = get_llm(temperature=0)
    tools = [calculator, save_note, get_note, list_notes, delete_note]
    checkpointer = MemorySaver()

    agent = create_react_agent(
        llm,
        tools,
        prompt=SystemMessage(content=MEMORY_AGENT_SYSTEM),
        checkpointer=checkpointer,
    )
    return agent
