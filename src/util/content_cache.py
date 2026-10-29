"""A bounded cache keyed by a hash of the inputs that produced each value."""

import hashlib
import json
import threading
from collections import OrderedDict


class ContentCache:
    """Values by the SHA-256 of their inputs, least recently used out first.

    It keeps only digests, never the inputs: an uploaded file is not retained after
    its request, and the cache's size is its entry count times a small value.
    """

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._items: OrderedDict[str, object] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(*inputs) -> str:
        encoded = json.dumps(inputs, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, key: str):
        with self._lock:
            if key not in self._items:
                return None
            self._items.move_to_end(key)
            return self._items[key]

    def put(self, key: str, value) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.maxsize:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._items)
