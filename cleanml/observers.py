from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PipelineObserver(ABC):
    @abstractmethod
    def on_event(self, event_type: str, data: dict[str, Any]) -> None:
        pass


class ConsoleLogger(PipelineObserver):
    def on_event(self, event_type: str, data: dict[str, Any]) -> None:
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
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
    
    def on_event(self, event_type: str, data: dict[str, Any]) -> None:
        self.events.append((event_type, data))
    
    def get_events(self) -> list[tuple[str, dict[str, Any]]]:
        return self.events
    
    def clear(self) -> None:
        self.events.clear()