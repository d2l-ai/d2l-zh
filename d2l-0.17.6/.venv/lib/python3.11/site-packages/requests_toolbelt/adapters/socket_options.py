# -*- coding: utf-8 -*-
"""The implementation of the SocketOptionsAdapter."""
import socket
import warnings
import sys

import requests
from requests import adapters

from .._compat import connection
from .._compat import poolmanager
from .. import exceptions as exc


class SocketOptionsAdapter(adapters.HTTPAdapter):
    """An adapter for requests that allows users to specify socket options.

    Since version 2.4.0 of requests, it is possible to specify a custom list
    of socket options that need to be set before establishing the connection.

    Example usage::

        >>> import socket
        >>> import requests
        >>> from requests_toolbelt.adapters import socket_options
        >>> s = requests.Session()
        >>> opts = [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 0)]
        >>> adapter = socket_options.SocketOptionsAdapter(socket_options=opts)
        >>> s.mount('http://', adapter)

    You can also take advantage of the list of default options on this class
    to keep using the original options in addition to your custom options. In
    that case, ``opts`` might look like::

        >>> opts = socket_options.SocketOptionsAdapter.default_options + opts

    """

    if connection is not None:
        default_options = getattr(
            connection.HTTPConnection,
            'default_socket_options',
            [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]
        )
    else:
        default_options = []
        warnings.warn(exc.RequestsVersionTooOld,
                      "This version of Requests is only compatible with a "
                      "version of urllib3 which is too old to support "
                      "setting options on a socket. This adapter is "
                      "functionally useless.")

    def __init__(self, **kwargs):
        self.socket_options = kwargs.pop('socket_options',
                                         self.default_options)

        super(SocketOptionsAdapter, self).__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        if requests.__build__ >= 0x020400:
            # NOTE(Ian): Perhaps we should raise a warning
            self.poolmanager = poolmanager.PoolManager(
                num_pools=connections,
                maxsize=maxsize,
                block=block,
                socket_options=self.socket_options
            )
        else:
            super(SocketOptionsAdapter, self).init_poolmanager(
                connections, maxsize, block
            )


class TCPKeepAliveAdapter(SocketOptionsAdapter):
    """An adapter for requests that turns on TCP Keep-Alive by default.

    The adapter sets 4 socket options:

    - ``SOL_SOCKET`` ``SO_KEEPALIVE`` - This turns on TCP Keep-Alive
    - ``IPPROTO_TCP`` ``TCP_KEEPINTVL`` 20 - Sets the keep alive interval
    - ``IPPROTO_TCP`` ``TCP_KEEPCNT`` 5 - Sets the number of keep alive probes
    - ``IPPROTO_TCP`` ``TCP_KEEPIDLE`` 60 - Sets the keep alive time if the
      socket library has the ``TCP_KEEPIDLE`` constant

    The latter three can be overridden by keyword arguments (respectively):

    - ``interval``
    - ``count``
    - ``idle``

    You can use this adapter like so::

       >>> from requests_toolbelt.adapters import socket_options
       >>> tcp = socket_options.TCPKeepAliveAdapter(idle=120, interval=10)
       >>> s = requests.Session()
       >>> s.mount('http://', tcp)

    """

    def __init__(self, **kwargs):
        socket_options = kwargs.pop('socket_options',
                                    SocketOptionsAdapter.default_options)
        idle = kwargs.pop('idle', 60)
        interval = kwargs.pop('interval', 20)
        count = kwargs.pop('count', 5)
        socket_options = socket_options + [
            (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        ]

        # NOTE(Ian): OSX does not have these constants defined, so we
        # set them conditionally.
        if getattr(socket, 'TCP_KEEPINTVL', None) is not None:
            socket_options += [(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL,
                                interval)]
        elif sys.platform == 'darwin':
            # On OSX, TCP_KEEPALIVE from netinet/tcp.h is not exported
            # by python's socket module
            TCP_KEEPALIVE = getattr(socket, 'TCP_KEEPALIVE', 0x10)
            socket_options += [(socket.IPPROTO_TCP, TCP_KEEPALIVE, interval)]

        if getattr(socket, 'TCP_KEEPCNT', None) is not None:
            socket_options += [(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, count)]

        if getattr(socket, 'TCP_KEEPIDLE', None) is not None:
            socket_options += [(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, idle)]

        super(TCPKeepAliveAdapter, self).__init__(
            socket_options=socket_options, **kwargs
        )
