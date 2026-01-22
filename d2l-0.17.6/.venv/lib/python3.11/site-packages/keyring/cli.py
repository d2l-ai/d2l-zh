"""Simple command line interface to get/set password from a keyring"""

from __future__ import annotations

import argparse
import getpass
import json
import sys

from . import (
    backend,
    completion,
    core,
    credentials,
    delete_password,
    get_credential,
    get_password,
    set_keyring,
    set_password,
)
from .util import platform_


class CommandLineTool:
    # Attributes set dynamically by the ArgumentParser
    keyring_path: str | None
    keyring_backend: str | None
    get_mode: str
    output_format: str
    operation: str
    service: str
    username: str

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "-p",
            "--keyring-path",
            dest="keyring_path",
            default=None,
            help="Path to the keyring backend",
        )
        self.parser.add_argument(
            "-b",
            "--keyring-backend",
            dest="keyring_backend",
            default=None,
            help="Name of the keyring backend",
        )
        self.parser.add_argument(
            "--list-backends",
            action="store_true",
            help="List keyring backends and exit",
        )
        self.parser.add_argument(
            "--disable", action="store_true", help="Disable keyring and exit"
        )
        self.parser._get_modes = ["password", "creds"]
        self.parser.add_argument(
            "--mode",
            choices=self.parser._get_modes,
            dest="get_mode",
            default="password",
            help="""
            Mode for 'get' operation.
            'password' requires a username and will return only the password.
            'creds' does not require a username and will return both the username and password separated by a newline.

            Default is 'password'
            """,
        )
        self.parser._output_formats = ["plain", "json"]
        self.parser.add_argument(
            "--output",
            choices=self.parser._output_formats,
            dest="output_format",
            default="plain",
            help="""
            Output format for 'get' operation.

            Default is 'plain'
            """,
        )
        self.parser._operations = ["get", "set", "del", "diagnose"]
        self.parser.add_argument(
            'operation',
            choices=self.parser._operations,
            nargs="?",
        )
        self.parser.add_argument(
            'service',
            nargs="?",
        )
        self.parser.add_argument(
            'username',
            nargs="?",
        )
        completion.install(self.parser)

    def run(self, argv):
        args = self.parser.parse_args(argv)
        vars(self).update(vars(args))

        if args.list_backends:
            for k in backend.get_all_keyring():
                print(k)
            return

        if args.disable:
            core.disable()
            return

        if args.operation == 'diagnose':
            self.diagnose()
            return

        self._check_args()
        self._load_spec_backend()
        method = getattr(self, f'do_{self.operation}', self.invalid_op)
        return method()

    def _check_args(self):
        needs_username = self.operation != 'get' or self.get_mode != 'creds'
        required = (['service'] + ['username'] * needs_username) * bool(self.operation)
        if any(getattr(self, param) is None for param in required):
            self.parser.error(f"{self.operation} requires {' and '.join(required)}")

    def do_get(self):
        credential = getattr(self, f'_get_{self.get_mode}')()
        if credential is None:
            raise SystemExit(1)
        getattr(self, f'_emit_{self.output_format}')(credential)

    def _emit_json(self, credential: credentials.Credential):
        print(json.dumps(credential._vars()))

    def _emit_plain(self, credential: credentials.Credential):
        for val in credential._vars().values():
            print(val)

    def _get_creds(self) -> credentials.Credential | None:
        return get_credential(self.service, self.username)

    def _get_password(self) -> credentials.Credential | None:
        password = get_password(self.service, self.username)
        return (
            credentials.AnonymousCredential(password) if password is not None else None
        )

    def do_set(self):
        password = self.input_password(
            f"Password for '{self.username}' in '{self.service}': "
        )
        set_password(self.service, self.username, password)

    def do_del(self):
        delete_password(self.service, self.username)

    def diagnose(self):
        config_root = core._config_path()
        if config_root.exists():
            print("config path:", config_root)
        else:
            print("config path:", config_root, "(absent)")
        print("data root:", platform_.data_root())

    def invalid_op(self):
        self.parser.error(f"Specify operation ({', '.join(self.parser._operations)}).")

    def _load_spec_backend(self):
        if self.keyring_backend is None:
            return

        try:
            if self.keyring_path:
                sys.path.insert(0, self.keyring_path)
            set_keyring(core.load_keyring(self.keyring_backend))
        except Exception as exc:
            # Tons of things can go wrong here:
            #   ImportError when using "fjkljfljkl"
            #   AttributeError when using "os.path.bar"
            #   TypeError when using "__builtins__.str"
            # So, we play on the safe side, and catch everything.
            self.parser.error(f"Unable to load specified keyring: {exc}")

    def input_password(self, prompt):
        """Retrieve password from input."""
        return self.pass_from_pipe() or getpass.getpass(prompt)

    @classmethod
    def pass_from_pipe(cls):
        """Return password from pipe if not on TTY, else False."""
        is_pipe = not sys.stdin.isatty()
        return is_pipe and cls.strip_last_newline(sys.stdin.read())

    @staticmethod
    def strip_last_newline(str):
        r"""Strip one last newline, if present.

        >>> CommandLineTool.strip_last_newline('foo')
        'foo'
        >>> CommandLineTool.strip_last_newline('foo\n')
        'foo'
        """
        slc = slice(-1 if str.endswith('\n') else None)
        return str[slc]


def main(argv=None):
    """Main command line interface."""

    if argv is None:
        argv = sys.argv[1:]

    cli = CommandLineTool()
    return cli.run(argv)


if __name__ == '__main__':
    sys.exit(main())
