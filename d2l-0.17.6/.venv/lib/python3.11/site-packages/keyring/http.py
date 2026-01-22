"""
urllib2.HTTPPasswordMgr object using the keyring, for use with the
urllib2.HTTPBasicAuthHandler.

usage:
    import urllib2
    handlers = [urllib2.HTTPBasicAuthHandler(PasswordMgr())]
    urllib2.install_opener(handlers)
    urllib2.urlopen(...)

This will prompt for a password if one is required and isn't already
in the keyring. Then, it adds it to the keyring for subsequent use.
"""

import getpass

from . import delete_password, get_password, set_password


class PasswordMgr:
    def get_username(self, realm, authuri):
        return getpass.getuser()

    def add_password(self, realm, authuri, password):
        user = self.get_username(realm, authuri)
        set_password(realm, user, password)

    def find_user_password(self, realm, authuri):
        user = self.get_username(realm, authuri)
        password = get_password(realm, user)
        if password is None:
            prompt = f'password for {user}@{realm} for {authuri}: '
            password = getpass.getpass(prompt)
            set_password(realm, user, password)
        return user, password

    def clear_password(self, realm, authuri):
        user = self.get_username(realm, authuri)
        delete_password(realm, user)
