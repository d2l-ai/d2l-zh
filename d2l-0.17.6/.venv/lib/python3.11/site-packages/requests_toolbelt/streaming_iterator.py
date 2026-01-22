# -*- coding: utf-8 -*-
"""

requests_toolbelt.streaming_iterator
====================================

This holds the implementation details for the :class:`StreamingIterator`. It
is designed for the case where you, the user, know the size of the upload but
need to provide the data as an iterator. This class will allow you to specify
the size and stream the data without using a chunked transfer-encoding.

"""
from requests.utils import super_len

from .multipart.encoder import CustomBytesIO, encode_with


class StreamingIterator(object):

    """
    This class provides a way of allowing iterators with a known size to be
    streamed instead of chunked.

    In requests, if you pass in an iterator it assumes you want to use
    chunked transfer-encoding to upload the data, which not all servers
    support well. Additionally, you may want to set the content-length
    yourself to avoid this but that will not work. The only way to preempt
    requests using a chunked transfer-encoding and forcing it to stream the
    uploads is to mimic a very specific interace. Instead of having to know
    these details you can instead just use this class. You simply provide the
    size and iterator and pass the instance of StreamingIterator to requests
    via the data parameter like so:

    .. code-block:: python

        from requests_toolbelt import StreamingIterator

        import requests

        # Let iterator be some generator that you already have and size be
        # the size of the data produced by the iterator

        r = requests.post(url, data=StreamingIterator(size, iterator))

    You can also pass file-like objects to :py:class:`StreamingIterator` in
    case requests can't determize the filesize itself. This is the case with
    streaming file objects like ``stdin`` or any sockets. Wrapping e.g. files
    that are on disk with ``StreamingIterator`` is unnecessary, because
    requests can determine the filesize itself.

    Naturally, you should also set the `Content-Type` of your upload
    appropriately because the toolbelt will not attempt to guess that for you.
    """

    def __init__(self, size, iterator, encoding='utf-8'):
        #: The expected size of the upload
        self.size = int(size)

        if self.size < 0:
            raise ValueError(
                'The size of the upload must be a positive integer'
                )

        #: Attribute that requests will check to determine the length of the
        #: body. See bug #80 for more details
        self.len = self.size

        #: Encoding the input data is using
        self.encoding = encoding

        #: The iterator used to generate the upload data
        self.iterator = iterator

        if hasattr(iterator, 'read'):
            self._file = iterator
        else:
            self._file = _IteratorAsBinaryFile(iterator, encoding)

    def read(self, size=-1):
        return encode_with(self._file.read(size), self.encoding)


class _IteratorAsBinaryFile(object):
    def __init__(self, iterator, encoding='utf-8'):
        #: The iterator used to generate the upload data
        self.iterator = iterator

        #: Encoding the iterator is using
        self.encoding = encoding

        # The buffer we use to provide the correct number of bytes requested
        # during a read
        self._buffer = CustomBytesIO()

    def _get_bytes(self):
        try:
            return encode_with(next(self.iterator), self.encoding)
        except StopIteration:
            return b''

    def _load_bytes(self, size):
        self._buffer.smart_truncate()
        amount_to_load = size - super_len(self._buffer)
        bytes_to_append = True

        while amount_to_load > 0 and bytes_to_append:
            bytes_to_append = self._get_bytes()
            amount_to_load -= self._buffer.append(bytes_to_append)

    def read(self, size=-1):
        size = int(size)
        if size == -1:
            return b''.join(self.iterator)

        self._load_bytes(size)
        return self._buffer.read(size)
