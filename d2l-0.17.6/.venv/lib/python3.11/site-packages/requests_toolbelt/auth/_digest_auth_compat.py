"""Provide a compatibility layer for requests.auth.HTTPDigestAuth."""
import requests


class _ThreadingDescriptor(object):
    def __init__(self, prop, default):
        self.prop = prop
        self.default = default

    def __get__(self, obj, objtype=None):
        return getattr(obj._thread_local, self.prop, self.default)

    def __set__(self, obj, value):
        setattr(obj._thread_local, self.prop, value)


class _HTTPDigestAuth(requests.auth.HTTPDigestAuth):
    init = _ThreadingDescriptor('init', True)
    last_nonce = _ThreadingDescriptor('last_nonce', '')
    nonce_count = _ThreadingDescriptor('nonce_count', 0)
    chal = _ThreadingDescriptor('chal', {})
    pos = _ThreadingDescriptor('pos', None)
    num_401_calls = _ThreadingDescriptor('num_401_calls', 1)


if requests.__build__ < 0x020800:
    HTTPDigestAuth = requests.auth.HTTPDigestAuth
else:
    HTTPDigestAuth = _HTTPDigestAuth
