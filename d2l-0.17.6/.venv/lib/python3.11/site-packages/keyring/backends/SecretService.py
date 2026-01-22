import logging
from contextlib import closing

from jaraco.context import ExceptionTrap

from .. import backend
from ..backend import KeyringBackend
from ..compat import properties
from ..credentials import SimpleCredential
from ..errors import (
    InitError,
    KeyringLocked,
    PasswordDeleteError,
)

try:
    import secretstorage
    import secretstorage.exceptions as exceptions
except ImportError:
    pass
except AttributeError:
    # See https://github.com/jaraco/keyring/issues/296
    pass

log = logging.getLogger(__name__)


class Keyring(backend.SchemeSelectable, KeyringBackend):
    """Secret Service Keyring"""

    appid = 'Python keyring library'

    @properties.classproperty
    def priority(cls) -> float:
        with ExceptionTrap() as exc:
            secretstorage.__name__  # noqa: B018
        if exc:
            raise RuntimeError("SecretStorage required")
        if secretstorage.__version_tuple__ < (3, 2):
            raise RuntimeError("SecretStorage 3.2 or newer required")
        try:
            with closing(secretstorage.dbus_init()) as connection:
                if not secretstorage.check_service_availability(connection):
                    raise RuntimeError(
                        "The Secret Service daemon is neither running nor "
                        "activatable through D-Bus"
                    )
        except exceptions.SecretStorageException as e:
            raise RuntimeError(f"Unable to initialize SecretService: {e}") from e
        return 5

    def get_preferred_collection(self):
        """If self.preferred_collection contains a D-Bus path,
        the collection at that address is returned. Otherwise,
        the default collection is returned.
        """
        bus = secretstorage.dbus_init()
        try:
            if hasattr(self, 'preferred_collection'):
                collection = secretstorage.Collection(bus, self.preferred_collection)
            else:
                collection = secretstorage.get_default_collection(bus)
        except exceptions.SecretStorageException as e:
            raise InitError(f"Failed to create the collection: {e}.") from e
        if collection.is_locked():
            collection.unlock()
            if collection.is_locked():  # User dismissed the prompt
                raise KeyringLocked("Failed to unlock the collection!")
        return collection

    def unlock(self, item):
        if hasattr(item, 'unlock'):
            item.unlock()
        if item.is_locked():  # User dismissed the prompt
            raise KeyringLocked('Failed to unlock the item!')

    def get_password(self, service, username):
        """Get password of the username for the service"""
        collection = self.get_preferred_collection()
        with closing(collection.connection):
            items = collection.search_items(self._query(service, username))
            for item in items:
                self.unlock(item)
                return item.get_secret().decode('utf-8')

    def set_password(self, service, username, password):
        """Set password for the username of the service"""
        collection = self.get_preferred_collection()
        attributes = self._query(service, username, application=self.appid)
        label = f"Password for '{username}' on '{service}'"
        with closing(collection.connection):
            collection.create_item(label, attributes, password, replace=True)

    def delete_password(self, service, username):
        """Delete the stored password (only the first one)"""
        collection = self.get_preferred_collection()
        with closing(collection.connection):
            items = collection.search_items(self._query(service, username))
            for item in items:
                return item.delete()
        raise PasswordDeleteError("No such password!")

    def get_credential(self, service, username):
        """Gets the first username and password for a service.
        Returns a Credential instance

        The username can be omitted, but if there is one, it will use get_password
        and return a SimpleCredential containing  the username and password
        Otherwise, it will return the first username and password combo that it finds.
        """
        scheme = self.schemes[self.scheme]
        query = self._query(service, username)
        collection = self.get_preferred_collection()

        with closing(collection.connection):
            items = collection.search_items(query)
            for item in items:
                self.unlock(item)
                username = item.get_attributes().get(scheme['username'])
                return SimpleCredential(username, item.get_secret().decode('utf-8'))
