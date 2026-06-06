from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PipelineObserver(ABC):
    """Base class for objects that listen to pipeline events."""

    @abstractmethod
    def on_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Handle a pipeline event.

        Args:
            event_type: Name of the event that occurred.
            data: Event details provided by the pipeline.
        """
        pass


class ConsoleLogger(PipelineObserver):
    """Print pipeline events to the console."""

    def on_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Print a readable message for a pipeline event.

        Args:
            event_type: Name of the event that occurred.
            data: Event details provided by the pipeline.
        """
        if event_type == "pipeline_started":
            print("[PIPELINE STARTED]")
        
        elif event_type == "pipeline_finished":
            print("[PIPELINE FINISHED]")
        
        elif event_type == "step_started":
            step_name = data.get("step_name", "UnknownStep")
            print(f"[START] {step_name}")
        
        elif event_type == "step_finished":
            step_name = data.get("step_name", "UnknownStep")
            print(f"[DONE] {step_name}")
        
        elif event_type == "error_occurred":
            error = data.get("error", "Unknown error")
            print(f"[ERROR] {error}")


class EventHistory(PipelineObserver):
    """Store pipeline events in memory for later inspection."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
    
    def on_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Record a pipeline event.

        Args:
            event_type: Name of the event that occurred.
            data: Event details provided by the pipeline.
        """
        self.events.append((event_type, data))
    
    def get_events(self) -> list[tuple[str, dict[str, Any]]]:
        """Return all recorded events.

        Returns:
            A list of ``(event_type, data)`` tuples.
        """
        return self.events
    
    def clear(self) -> None:
        """Remove all recorded events."""
        self.events.clear()
