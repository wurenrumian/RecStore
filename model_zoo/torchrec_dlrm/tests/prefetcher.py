import queue
import threading
from typing import Dict, Optional, Tuple

import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class PrefetchingIterator:
    def __init__(self, dataloader, ebc_module, prefetch_count: int = 2):
        """
        dataloader: base DataLoader yielding (dense, KJT, labels)
        ebc_module: RecStoreEmbeddingBagCollection instance
        prefetch_count: queue depth
        NOTE: Must call restart() at each new epoch.
        """
        self._dataloader = dataloader
        self._prefetch_count = prefetch_count
        self._thread: Optional[threading.Thread] = None
        self._queue: "queue.Queue[Optional[Tuple[torch.Tensor, KeyedJaggedTensor, torch.Tensor, Dict[str, object]]]]" = queue.Queue(maxsize=self._prefetch_count)
        self._stop = False
        self._exhausted = False
        self._producer_error: Optional[BaseException] = None
        self._iter = iter(self._dataloader)
        self._start_thread()

    def _producer(self):
        try:
            while not self._stop:
                batch = next(self._iter)
                dense, sparse, labels = batch
                # Strict-step mode keeps producer-side batch preparation, but
                # it must not issue embedding prefetches for a future step.
                self._queue.put((dense, sparse, labels, {}))
        except StopIteration:
            self._exhausted = True
            self._queue.put(None)
        except Exception as e:
            self._producer_error = e
            self._exhausted = True
            self._queue.put(None)

    def _start_thread(self):
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()

    def restart(self):
        """Restart iteration for a new epoch."""
        self.stop(join=True)
        self._queue = queue.Queue(maxsize=self._prefetch_count)
        self._stop = False
        self._exhausted = False
        self._producer_error = None
        self._iter = iter(self._dataloader)
        self._start_thread()

    def __iter__(self):
        return self

    def __next__(self):
        if self._producer_error is not None:
            raise RuntimeError("PrefetchingIterator producer failed") from self._producer_error
        if self._exhausted and self._queue.empty():
            raise StopIteration

        item = self._queue.get()
        if item is None:
            if self._producer_error is not None:
                raise RuntimeError("PrefetchingIterator producer failed") from self._producer_error
            self._exhausted = True
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            raise StopIteration

        dense, sparse, labels, handles = item
        return dense, sparse, labels, handles

    def stop(self, join: bool = False):
        self._stop = True
        try:
            while True:
                self._queue.get_nowait()
        except Exception:
            pass
        if self._thread and self._thread.is_alive() and join:
            self._thread.join(timeout=1)
            if self._thread.is_alive():
                raise RuntimeError("PrefetchingIterator producer thread did not stop cleanly")
