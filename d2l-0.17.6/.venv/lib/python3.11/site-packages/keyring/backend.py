"""
Keyring implementation support
"""

from __future__ import annotations

import abc
import copy
import functools
import logging
import operator
import os
import typing
import warnings

from jaraco.context import ExceptionTrap
from jaraco.functools import once

from . import credentials, errors, util
from .compat import properties
from .compat.py312 import metadata

log = logging.getLogger(__name__)


by_priority = operator.attrgetter('priority')
_limit: typing.Callable[[KeyringBackend], bool] | None = None


class KeyringBackendMeta(abc.ABCMeta):
    """
    Specialized subclass behavior.

    Keeps a registry of all (non-abstract) types.

    Wraps set_password to validate the username.
    """

    def __init__(cls, name, bases, dict):
        super().__init__(name, bases, dict)
        cls._register()
        cls._validate_username_in_set_password()

    def _register(cls):
        if not hasattr(cls, '_classes'):
            cls._classes = set()
        classes = cls._classes
        if not cls.__abstractmethods__:
            classes.add(cls)

    def _validate_username_in_set_password(cls):
        """
        Wrap ``set_password`` such to validate the passed username.
        """
        orig = cls.set_password

        @functools.wraps(orig)
        def wrapper(self, system, username, *args, **kwargs):
            self._validate_username(username)
            return orig(self, system, username, *args, **kwargs)

        cls.set_password = wrapper


class KeyringBackend(metaclass=KeyringBackendMeta):
    """The abstract base class of the keyring, every backend must implement
    this interface.
    """

    def __init__(self):
        self.set_properties_from_env()

    @properties.classproperty
    def priority(self) -> float:
        """
        Each backend class must supply a priority, a number (float or integer)
        indicating the priority of the backend relative to all other backends.
        The priority need not be static -- it may (and should) vary based
        attributes of the environment in which is runs (platform, available
        packages, etc.).

        A higher number indicates a higher priority. The priority should raise
        a RuntimeError with a message indicating the underlying cause if the
        backend is not suitable for the current environment.

        As a rule of thumb, a priority between zero but less than one is
        suitable, but a priority of one or greater is recommended.
        """
        raise NotImplementedError

    # Python 3.8 compatibility
    passes = ExceptionTrap().passes

    @properties.classproperty
    @passes
    def viable(cls):
        cls.priority  # noqa: B018

    @classmethod
    def get_viable_backends(
        cls: type[KeyringBackend],
    ) -> filter[type[KeyringBackend]]:
        """
        Return all subclasses deemed viable.
        """
        return filter(operator.attrgetter('viable'), cls._classes)

    @properties.classproperty
    def name(cls) -> str:
        """
        The keyring name, suitable for display.

        The name is derived from module and class name.
        """
        parent, sep, mod_name = cls.__module__.rpartition('.')
        mod_name = mod_name.replace('_', ' ')
        # mypy doesn't see `cls` is `type[Self]`, might be fixable in jaraco.classes
        return ' '.join([mod_name, cls.__name__])  # type: ignore[attr-defined]

    def __str__(self) -> str:
        keyring_class = type(self)
        return f"{keyring_class.__module__}.{keyring_class.__name__} (priority: {keyring_class.priority:g})"

    @abc.abstractmethod
    def get_password(self, service: str, username: str) -> str | None:
        """Get password of the username for the service"""
        return None

    def _validate_username(self, username: str) -> None:
        """
        Ensure the username is not empty.
        """
        if not username:
            warnings.warn(
                "Empty usernames are deprecated. See #668",
                DeprecationWarning,
                stacklevel=3,
            )
            # raise ValueError("Username cannot be empty")

    @abc.abstractmethod
    def set_password(self, service: str, username: str, password: str) -> None:
        """Set password for the username of the service.

        If the backend cannot store passwords, raise
        PasswordSetError.
        """
        raise errors.PasswordSetError("reason")

    # for backward-compatibility, don't require a backend to implement
    #  delete_password
    # @abc.abstractmethod
    def delete_password(self, service: str, username: str) -> None:
        """Delete the password for the username of the service.

        If the backend cannot delete passwords, raise
        PasswordDeleteError.
        """
        raise errors.PasswordDeleteError("reason")

    # for backward-compatibility, don't require a backend to implement
    #  get_credential
    # @abc.abstractmethod
    def get_credential(
        self,
        service: str,
        username: str | None,
    ) -> credentials.Credential | None:
        """Gets the username and password for the service.
        Returns a Credential instance.

        The *username* argument is optional and may be omitted by
        the caller or ignored by the backend. Callers must use the
        returned username.
        """
        # The default implementation requires a username here.
        if username is not None:
            password = self.get_password(service, username)
            if password is not None:
                return credentials.SimpleCredential(username, password)
        return None

    def set_properties_from_env(self) -> None:
        """For all KEYRING_PROPERTY_* env var, set that property."""

        def parse(item: tuple[str, str]):
            key, value = item
            pre, sep, name = key.partition('KEYRING_PROPERTY_')
            return sep and (name.lower(), value)

        props: filter[tuple[str, str]] = filter(None, map(parse, os.environ.items()))
        for name, value in props:
            setattr(self, name, value)

    def with_properties(self, **kwargs: typing.Any) -> KeyringBackend:
        alt = copy.copy(self)
        vars(alt).update(kwargs)
        return alt


class Crypter:
    """Base class providing encryption and decryption"""

    @abc.abstractmethod
    def encrypt(self, value):
        """Encrypt the value."""
        pass

    @abc.abstractmethod
    def decrypt(self, value):
        """Decrypt the value."""
        pass


class NullCrypter(Crypter):
    """A crypter that does nothing"""

    def encrypt(self, value):
        return value

    def decrypt(self, value):
        return value


def _load_plugins() -> None:
    """
    Locate all setuptools entry points by the name 'keyring backends'
    and initialize them.
    Any third-party library may register an entry point by adding the
    following to their setup.cfg::

        [options.entry_points]
        keyring.backends =
            plugin_name = mylib.mymodule:initialize_func

    `plugin_name` can be anything, and is only used to display the name
    of the plugin at initialization time.

    `initialize_func` is optional, but will be invoked if callable.
    """
    for ep in metadata.entry_points(group='keyring.backends'):
        try:
            log.debug('Loading %s', ep.name)
            init_func = ep.load()
            if callable(init_func):
                init_func()
        except Exception:
            log.exception(f"Error initializing plugin {ep}.")


@once
def get_all_keyring() -> list[KeyringBackend]:
    """
    Return a list of all implemented keyrings that can be constructed without
    parameters.
    """
    _load_plugins()
    viable_classes = KeyringBackend.get_viable_backends()
    rings = util.suppress_exceptions(viable_classes, exceptions=TypeError)
    return list(rings)


class SchemeSelectable:
    """
    Allow a backend to select different "schemes" for the
    username and service.

    >>> backend = SchemeSelectable()
    >>> backend._query('contoso', 'alice')
    {'username': 'alice', 'service': 'contoso'}
    >>> backend._query('contoso')
    {'service': 'contoso'}
    >>> backend.scheme = 'KeePassXC'
    >>> backend._query('contoso', 'alice')
    {'UserName': 'alice', 'Title': 'contoso'}
    >>> backend._query('contoso', 'alice', foo='bar')
    {'UserName': 'alice', 'Title': 'contoso', 'foo': 'bar'}
    """

    scheme = 'default'
    schemes = dict(
        default=dict(username='username', service='service'),
        KeePassXC=dict(username='UserName', service='Title'),
    )

    def _query(
        self, service: str, username: str | None = None, **base: typing.Any
    ) -> dict[str, str]:
        scheme = self.schemes[self.scheme]
        return dict(
            {
                scheme['username']: username,
                scheme['service']: service,
            }
            if username is not None
            else {
                scheme['service']: service,
            },
            **base,
        )
