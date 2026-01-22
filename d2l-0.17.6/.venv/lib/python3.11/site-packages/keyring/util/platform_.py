import os
import pathlib
import platform


def _data_root_Windows():
    release, version, csd, ptype = platform.win32_ver()
    root = pathlib.Path(
        os.environ.get('LOCALAPPDATA', os.environ.get('ProgramData', '.'))
    )
    return root / 'Python Keyring'


def _data_root_Linux():
    """
    Use freedesktop.org Base Dir Specification to determine storage
    location.
    """
    fallback = pathlib.Path.home() / '.local/share'
    root = os.environ.get('XDG_DATA_HOME', None) or fallback
    return pathlib.Path(root, 'python_keyring')


_config_root_Windows = _data_root_Windows


def _config_root_Linux():
    """
    Use freedesktop.org Base Dir Specification to determine config
    location.
    """
    fallback = pathlib.Path.home() / '.config'
    key = 'XDG_CONFIG_HOME'
    root = os.environ.get(key, None) or fallback
    return pathlib.Path(root, 'python_keyring')


# by default, use Unix convention
data_root = globals().get('_data_root_' + platform.system(), _data_root_Linux)
config_root = globals().get('_config_root_' + platform.system(), _config_root_Linux)
