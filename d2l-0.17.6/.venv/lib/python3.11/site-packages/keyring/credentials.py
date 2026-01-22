from __future__ import annotations

import abc
import os


class Credential(metaclass=abc.ABCMeta):
    """Abstract class to manage credentials"""

    @abc.abstractproperty
    def username(self) -> str: ...

    @abc.abstractproperty
    def password(self) -> str: ...

    def _vars(self) -> dict[str, str]:
        return dict(username=self.username, password=self.password)


class SimpleCredential(Credential):
    """Simple credentials implementation"""

    def __init__(self, username: str, password: str):
        self._username = username
        self._password = password

    @property
    def username(self) -> str:
        return self._username

    @property
    def password(self) -> str:
        return self._password


class AnonymousCredential(SimpleCredential):
    def __init__(self, password: str):
        self._password = password

    @property
    def username(self) -> str:
        raise ValueError("Anonymous credential has no username")

    def _vars(self) -> dict[str, str]:
        return dict(password=self.password)


class EnvironCredential(Credential):
    """
    Source credentials from environment variables.

    Actual sourcing is deferred until requested.

    Supports comparison by equality.

    >>> e1 = EnvironCredential('a', 'b')
    >>> e2 = EnvironCredential('a', 'b')
    >>> e3 = EnvironCredential('a', 'c')
    >>> e1 == e2
    True
    >>> e2 == e3
    False
    """

    def __init__(self, user_env_var: str, pwd_env_var: str):
        self.user_env_var = user_env_var
        self.pwd_env_var = pwd_env_var

    def __eq__(self, other: object) -> bool:
        return vars(self) == vars(other)

    def _get_env(self, env_var: str) -> str:
        """Helper to read an environment variable"""
        value = os.environ.get(env_var)
        if not value:
            raise ValueError(f'Missing environment variable:{env_var}')
        return value

    @property
    def username(self) -> str:
        return self._get_env(self.user_env_var)

    @property
    def password(self) -> str:
        return self._get_env(self.pwd_env_var)
