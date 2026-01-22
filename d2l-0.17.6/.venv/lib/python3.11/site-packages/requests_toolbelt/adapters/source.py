# -*- coding: utf-8 -*-
"""
requests_toolbelt.source_adapter
================================

This file contains an implementation of the SourceAddressAdapter originally
demonstrated on the Requests GitHub page.
"""
from requests.adapters import HTTPAdapter

from .._compat import poolmanager, basestring


class SourceAddressAdapter(HTTPAdapter):
    """
    A Source Address Adapter for Python Requests that enables you to choose the
    local address to bind to. This allows you to send your HTTP requests from a
    specific interface and IP address.

    Two address formats are accepted. The first is a string: this will set the
    local IP address to the address given in the string, and will also choose a
    semi-random high port for the local port number.

    The second is a two-tuple of the form (ip address, port): for example,
    ``('10.10.10.10', 8999)``. This will set the local IP address to the first
    element, and the local port to the second element. If ``0`` is used as the
    port number, a semi-random high port will be selected.

    .. warning:: Setting an explicit local port can have negative interactions
                 with connection-pooling in Requests: in particular, it risks
                 the possibility of getting "Address in use" errors. The
                 string-only argument is generally preferred to the tuple-form.

    Example usage:

    .. code-block:: python

        import requests
        from requests_toolbelt.adapters.source import SourceAddressAdapter

        s = requests.Session()
        s.mount('http://', SourceAddressAdapter('10.10.10.10'))
        s.mount('https://', SourceAddressAdapter(('10.10.10.10', 8999)))
    """
    def __init__(self, source_address, **kwargs):
        if isinstance(source_address, basestring):
            self.source_address = (source_address, 0)
        elif isinstance(source_address, tuple):
            self.source_address = source_address
        else:
            raise TypeError(
                "source_address must be IP address string or (ip, port) tuple"
            )

        super(SourceAddressAdapter, self).__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = poolmanager.PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            source_address=self.source_address)

    def proxy_manager_for(self, *args, **kwargs):
        kwargs['source_address'] = self.source_address
        return super(SourceAddressAdapter, self).proxy_manager_for(
            *args, **kwargs)
