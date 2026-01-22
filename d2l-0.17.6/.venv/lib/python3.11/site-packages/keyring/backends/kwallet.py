import contextlib
import os
import sys

from ..backend import KeyringBackend
from ..compat import properties
from ..credentials import SimpleCredential
from ..errors import InitError, KeyringLocked, PasswordDeleteError, PasswordSetError

try:
    import dbus
    from dbus.mainloop.glib import DBusGMainLoop
except ImportError:
    pass
except AttributeError:
    # See https://github.com/jaraco/keyring/issues/296
    pass


def _id_from_argv():
    """
    Safely infer an app id from sys.argv.
    """
    allowed = AttributeError, IndexError, TypeError
    with contextlib.suppress(allowed):
        return sys.argv[0]


class DBusKeyring(KeyringBackend):
    """
    KDE KWallet 5 via D-Bus
    """

    appid = _id_from_argv() or 'Python keyring library'
    wallet = None
    bus_name = 'org.kde.kwalletd5'
    object_path = '/modules/kwalletd5'

    @properties.classproperty
    def priority(cls) -> float:
        if 'dbus' not in globals():
            raise RuntimeError('python-dbus not installed')
        try:
            bus = dbus.SessionBus(mainloop=DBusGMainLoop())
        except dbus.DBusException as exc:
            raise RuntimeError(exc.get_dbus_message()) from exc
        if not (
            bus.name_has_owner(cls.bus_name)
            and cls.bus_name in bus.list_activatable_names()
        ):
            raise RuntimeError(
                "The KWallet daemon is neither running nor activatable through D-Bus"
            )
        if "KDE" in os.getenv("XDG_CURRENT_DESKTOP", "").split(":"):
            return 5.1
        return 4.9

    def __init__(self, *arg, **kw):
        super().__init__(*arg, **kw)
        self.handle = -1

    def _migrate(self, service):
        old_folder = 'Python'
        entry_list = []
        if self.iface.hasFolder(self.handle, old_folder, self.appid):
            entry_list = self.iface.readPasswordList(
                self.handle, old_folder, '*@*', self.appid
            )

            for entry in entry_list.items():
                key = entry[0]
                password = entry[1]

                username, service = key.rsplit('@', 1)
                ret = self.iface.writePassword(
                    self.handle, service, username, password, self.appid
                )
                if ret == 0:
                    self.iface.removeEntry(self.handle, old_folder, key, self.appid)

            entry_list = self.iface.readPasswordList(
                self.handle, old_folder, '*', self.appid
            )
            if not entry_list:
                self.iface.removeFolder(self.handle, old_folder, self.appid)

    def connected(self, service):
        if self.handle >= 0:
            if self.iface.isOpen(self.handle):
                return True

        bus = dbus.SessionBus(mainloop=DBusGMainLoop())
        wId = 0
        try:
            remote_obj = bus.get_object(self.bus_name, self.object_path)
            self.iface = dbus.Interface(remote_obj, 'org.kde.KWallet')
            self.handle = self.iface.open(self.iface.networkWallet(), wId, self.appid)
        except dbus.DBusException as e:
            raise InitError(f'Failed to open keyring: {e}.') from e

        if self.handle < 0:
            return False
        self._migrate(service)
        return True

    def get_password(self, service, username):
        """Get password of the username for the service"""
        if not self.connected(service):
            # the user pressed "cancel" when prompted to unlock their keyring.
            raise KeyringLocked("Failed to unlock the keyring!")
        if not self.iface.hasEntry(self.handle, service, username, self.appid):
            return None
        password = self.iface.readPassword(self.handle, service, username, self.appid)
        return str(password)

    def get_credential(self, service, username):
        """Gets the first username and password for a service.
        Returns a Credential instance

        The username can be omitted, but if there is one, it will forward to
        get_password.
        Otherwise, it will return the first username and password combo that it finds.
        """
        if username is not None:
            return super().get_credential(service, username)

        if not self.connected(service):
            # the user pressed "cancel" when prompted to unlock their keyring.
            raise KeyringLocked("Failed to unlock the keyring!")

        for username in self.iface.entryList(self.handle, service, self.appid):
            password = self.iface.readPassword(
                self.handle, service, username, self.appid
            )
            return SimpleCredential(str(username), str(password))

    def set_password(self, service, username, password):
        """Set password for the username of the service"""
        if not self.connected(service):
            # the user pressed "cancel" when prompted to unlock their keyring.
            raise PasswordSetError("Cancelled by user")
        self.iface.writePassword(self.handle, service, username, password, self.appid)

    def delete_password(self, service, username):
        """Delete the password for the username of the service."""
        if not self.connected(service):
            # the user pressed "cancel" when prompted to unlock their keyring.
            raise PasswordDeleteError("Cancelled by user")
        if not self.iface.hasEntry(self.handle, service, username, self.appid):
            raise PasswordDeleteError("Password not found")
        self.iface.removeEntry(self.handle, service, username, self.appid)


class DBusKeyringKWallet4(DBusKeyring):
    """
    KDE KWallet 4 via D-Bus
    """

    bus_name = 'org.kde.kwalletd'
    object_path = '/modules/kwalletd'

    @properties.classproperty
    def priority(cls):
        return super().priority - 1
