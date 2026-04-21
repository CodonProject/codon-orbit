from typing import TYPE_CHECKING, Any, List, Callable, Union, Dict
from dataclasses import dataclass

if TYPE_CHECKING:
    from .engine import Engine

def once(fn: Callable) -> Callable:
    setattr(fn, '__once__', True)
    return fn

@dataclass
class Event:
    info: str
    sender: str
    data: Any
    engine: 'Engine'

class EventBroker:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = {}
        self._once_subscribers: Dict[str, List[Callable[[Event], None]]] = {}

    @staticmethod
    def _as_list(callbacks):
        return callbacks if isinstance(callbacks, list) else [callbacks]

    def subscribe(self, event_name: str, callbacks: Union[Callable, List[Callable]]) -> 'EventBroker':
        bucket = self._subscribers.setdefault(event_name, [])
        for cb in self._as_list(callbacks):
            if callable(cb) and cb not in bucket:
                bucket.append(cb)
        return self

    def subscribe_once(self, event_name: str, callbacks: Union[Callable, List[Callable]]) -> 'EventBroker':
        bucket = self._once_subscribers.setdefault(event_name, [])
        for cb in self._as_list(callbacks):
            if callable(cb) and cb not in bucket:
                bucket.append(cb)
        return self

    def unsubscribe(self, event_name: str, callbacks: Union[Callable, List[Callable]]) -> 'EventBroker':
        for cb in self._as_list(callbacks):
            if event_name in self._subscribers and cb in self._subscribers[event_name]:
                self._subscribers[event_name].remove(cb)
            if event_name in self._once_subscribers and cb in self._once_subscribers[event_name]:
                self._once_subscribers[event_name].remove(cb)
        return self

    def emit(self, event: Event) -> None:
        for cb in self._subscribers.get(event.info, ()):
            cb(event)
        once = self._once_subscribers.pop(event.info, None)
        if once:
            for cb in once:
                cb(event)