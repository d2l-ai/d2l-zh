# -*- coding: utf-8 -*-
"""
requests-toolbelt.adapters
==========================

See https://toolbelt.readthedocs.io/ for documentation

:copyright: (c) 2014 by Ian Cordasco and Cory Benfield
:license: Apache v2.0, see LICENSE for more details
"""

from .ssl import SSLAdapter
from .source import SourceAddressAdapter

__all__ = ['SSLAdapter', 'SourceAddressAdapter']
