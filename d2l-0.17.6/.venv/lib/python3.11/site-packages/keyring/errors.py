import sys
import warnings


class KeyringError(Exception):
    """Base class for exceptions in keyring"""


class PasswordSetError(KeyringError):
    """Raised when the password can't be set."""


class PasswordDeleteError(KeyringError):
    """Raised when the password can't be deleted."""


class InitError(KeyringError):
    """Raised when the keyring could not be initialised"""


class KeyringLocked(KeyringError):
    """Raised when the keyring failed unlocking"""


class NoKeyringError(KeyringError, RuntimeError):
    """Raised when there is no keyring backend"""


class ExceptionRaisedContext:
    """
    An exception-trapping context that indicates whether an exception was
    raised.
    """

    def __init__(self, ExpectedException=Exception):
        warnings.warn(
            "ExceptionRaisedContext is deprecated; use `jaraco.context.ExceptionTrap`",
            DeprecationWarning,
            stacklevel=2,
        )
        self.ExpectedException = ExpectedException
        self.exc_info = None

    def __enter__(self):
        self.exc_info = object.__new__(ExceptionInfo)
        return self.exc_info

    def __exit__(self, *exc_info):
        self.exc_info.__init__(*exc_info)
        return self.exc_info.type and issubclass(
            self.exc_info.type, self.ExpectedException
        )


class ExceptionInfo:
    def __init__(self, *info):
        if not info:
            info = sys.exc_info()
        self.type, self.value, _ = info

    def __bool__(self):
        """
        Return True if an exception occurred
        """
        return bool(self.type)

    __nonzero__ = __bool__
