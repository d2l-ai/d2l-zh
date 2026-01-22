# -*- coding: utf-8 -*-
"""

requests_toolbelt.ssl_adapter
=============================

This file contains an implementation of the SSLAdapter originally demonstrated
in this blog post:
https://lukasa.co.uk/2013/01/Choosing_SSL_Version_In_Requests/

"""
import requests

from requests.adapters import HTTPAdapter

from .._compat import poolmanager


class SSLAdapter(HTTPAdapter):
    """
    A HTTPS Adapter for Python Requests that allows the choice of the SSL/TLS
    version negotiated by Requests. This can be used either to enforce the
    choice of high-security TLS versions (where supported), or to work around
    misbehaving servers that fail to correctly negotiate the default TLS
    version being offered.

    Example usage:

        >>> import requests
        >>> import ssl
        >>> from requests_toolbelt import SSLAdapter
        >>> s = requests.Session()
        >>> s.mount('https://', SSLAdapter(ssl.PROTOCOL_TLSv1))

    You can replace the chosen protocol with any that are available in the
    default Python SSL module. All subsequent requests that match the adapter
    prefix will use the chosen SSL version instead of the default.

    This adapter will also attempt to change the SSL/TLS version negotiated by
    Requests when using a proxy. However, this may not always be possible:
    prior to Requests v2.4.0 the adapter did not have access to the proxy setup
    code. In earlier versions of Requests, this adapter will not function
    properly when used with proxies.
    """

    __attrs__ = HTTPAdapter.__attrs__ + ['ssl_version']

    def __init__(self, ssl_version=None, **kwargs):
        self.ssl_version = ssl_version

        super(SSLAdapter, self).__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = poolmanager.PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            ssl_version=self.ssl_version)

    if requests.__build__ >= 0x020400:
        # Earlier versions of requests either don't have this method or, worse,
        # don't allow passing arbitrary keyword arguments. As a result, only
        # conditionally define this method.
        def proxy_manager_for(self, *args, **kwargs):
            kwargs['ssl_version'] = self.ssl_version
            return super(SSLAdapter, self).proxy_manager_for(*args, **kwargs)
