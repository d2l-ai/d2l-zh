# -*- coding: utf-8 -*-
"""
requests-toolbelt
=================

See https://toolbelt.readthedocs.io/ for documentation

:copyright: (c) 2014 by Ian Cordasco and Cory Benfield
:license: Apache v2.0, see LICENSE for more details
"""

from .adapters import SSLAdapter, SourceAddressAdapter
from .auth.guess import GuessAuth
from .multipart import (
    MultipartEncoder, MultipartEncoderMonitor, MultipartDecoder,
    ImproperBodyPartContentException, NonMultipartContentTypeException
    )
from .streaming_iterator import StreamingIterator
from .utils.user_agent import user_agent

__title__ = 'requests-toolbelt'
__authors__ = 'Ian Cordasco, Cory Benfield'
__license__ = 'Apache v2.0'
__copyright__ = 'Copyright 2014 Ian Cordasco, Cory Benfield'
__version__ = '1.0.0'
__version_info__ = tuple(int(i) for i in __version__.split('.'))

__all__ = [
    'GuessAuth', 'MultipartEncoder', 'MultipartEncoderMonitor',
    'MultipartDecoder', 'SSLAdapter', 'SourceAddressAdapter',
    'StreamingIterator', 'user_agent', 'ImproperBodyPartContentException',
    'NonMultipartContentTypeException', '__title__', '__authors__',
    '__license__', '__copyright__', '__version__', '__version_info__',
]
