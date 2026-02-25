from typing import Callable, Dict, List
from collections import defaultdict


class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Callable):
        self._subscribers[event_type].append(handler)

    def publish(self, event_type: str, payload: dict):
        handlers = self._subscribers.get(event_type, [])
        for handler in handlers:
            handler(payload)


# Singleton instance (safe for modular monolith)
event_bus = EventBus()