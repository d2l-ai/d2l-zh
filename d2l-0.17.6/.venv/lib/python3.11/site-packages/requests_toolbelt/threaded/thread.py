"""Module containing the SessionThread class."""
import threading
import uuid

import requests.exceptions as exc

from .._compat import queue


class SessionThread(object):
    def __init__(self, initialized_session, job_queue, response_queue,
                 exception_queue):
        self._session = initialized_session
        self._jobs = job_queue
        self._create_worker()
        self._responses = response_queue
        self._exceptions = exception_queue

    def _create_worker(self):
        self._worker = threading.Thread(
            target=self._make_request,
            name=uuid.uuid4(),
        )
        self._worker.daemon = True
        self._worker._state = 0
        self._worker.start()

    def _handle_request(self, kwargs):
        try:
            response = self._session.request(**kwargs)
        except exc.RequestException as e:
            self._exceptions.put((kwargs, e))
        else:
            self._responses.put((kwargs, response))
        finally:
            self._jobs.task_done()

    def _make_request(self):
        while True:
            try:
                kwargs = self._jobs.get_nowait()
            except queue.Empty:
                break

            self._handle_request(kwargs)

    def is_alive(self):
        """Proxy to the thread's ``is_alive`` method."""
        return self._worker.is_alive()

    def join(self):
        """Join this thread to the master thread."""
        self._worker.join()
