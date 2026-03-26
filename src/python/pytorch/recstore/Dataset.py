import threading
import queue
from typing import Dict, Tuple, Optional, Callable, Any
import torch

class RecStoreDataset:
    def __init__(
        self,
        dataloader,
        client,
        key_extractor: Callable[[Any], Dict[str, torch.Tensor]],
        prefetch_count: int = 2
    ):
        self._dataloader = dataloader
        self._client = client
        self._key_extractor = key_extractor
        self._prefetch_count = prefetch_count
        self._thread: Optional[threading.Thread] = None
        self._queue = queue.Queue(maxsize=self._prefetch_count)
        self._stop = False
        self._exhausted = False
        self._iter = iter(self._dataloader)
        self._start_thread()

    def _producer(self):
        try:
            while not self._stop:
                batch = next(self._iter)
                ids_map = self._key_extractor(batch)
                handles = {}
                for key, ids in ids_map.items():
                    if ids.numel() > 0:
                        h = self._client.prefetch(ids)
                        handles[key] = h
                self._queue.put((batch, handles))
        except StopIteration:
            self._exhausted = True
            self._queue.put(None)
        except Exception as e:
            print(f"[RecStoreDataset] Error: {e}")
            self._queue.put(None)

    def _start_thread(self):
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()

    def restart(self):
        self.stop(join=True)
        self._queue = queue.Queue(maxsize=self._prefetch_count)
        self._stop = False
        self._exhausted = False
        self._iter = iter(self._dataloader)
        self._start_thread()

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is None:
            if not self._exhausted:
                self._exhausted = True
            raise StopIteration
        return item

    def stop(self, join: bool = False):
        self._stop = True
        try:
            while True:
                self._queue.get_nowait()
        except Exception:
            pass
        if self._thread and self._thread.is_alive() and join:
            self._thread.join(timeout=1)
