"""
Keyring Chainer - iterates over other viable backends to
discover passwords in each.
"""

from .. import backend
from ..compat import properties
from . import fail


class ChainerBackend(backend.KeyringBackend):
    """
    >>> ChainerBackend()
    <keyring.backends.chainer.ChainerBackend object at ...>
    """

    # override viability as 'priority' cannot be determined
    # until other backends have been constructed
    viable = True

    @properties.classproperty
    def priority(cls) -> float:
        """
        If there are backends to chain, high priority
        Otherwise very low priority since our operation when empty
        is the same as null.
        """
        return 10 if len(cls.backends) > 1 else (fail.Keyring.priority - 1)

    @properties.classproperty
    def backends(cls):
        """
        Discover all keyrings for chaining.
        """

        def allow(keyring):
            limit = backend._limit or bool
            return (
                not isinstance(keyring, ChainerBackend)
                and limit(keyring)
                and keyring.priority > 0
            )

        allowed = filter(allow, backend.get_all_keyring())
        return sorted(allowed, key=backend.by_priority, reverse=True)

    def get_password(self, service, username):
        for keyring in self.backends:
            password = keyring.get_password(service, username)
            if password is not None:
                return password

    def set_password(self, service, username, password):
        for keyring in self.backends:
            try:
                return keyring.set_password(service, username, password)
            except NotImplementedError:
                pass

    def delete_password(self, service, username):
        for keyring in self.backends:
            try:
                return keyring.delete_password(service, username)
            except NotImplementedError:
                pass

    def get_credential(self, service, username):
        for keyring in self.backends:
            credential = keyring.get_credential(service, username)
            if credential is not None:
                return credential
