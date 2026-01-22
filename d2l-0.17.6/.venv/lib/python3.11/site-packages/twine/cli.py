# Copyright 2013 Donald Stufft
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging.config
import sys
from typing import Any, List, Tuple

if sys.version_info >= (3, 10):
    import importlib.metadata as importlib_metadata
else:
    import importlib_metadata

import rich
import rich.highlighter
import rich.logging
import rich.theme

import twine

args = argparse.Namespace()


def configure_output() -> None:
    # Configure the global Console, available via rich.get_console().
    # https://rich.readthedocs.io/en/latest/reference/init.html
    # https://rich.readthedocs.io/en/latest/console.html
    rich.reconfigure(
        # Setting force_terminal makes testing easier by ensuring color codes. This
        # could be based on FORCE_COLORS or PY_COLORS in os.environ, since Rich
        # doesn't support that (https://github.com/Textualize/rich/issues/343).
        force_terminal=True,
        no_color=getattr(args, "no_color", False),
        highlight=False,
        theme=rich.theme.Theme(
            {
                "logging.level.debug": "green",
                "logging.level.info": "blue",
                "logging.level.warning": "yellow",
                "logging.level.error": "red",
                "logging.level.critical": "reverse red",
            }
        ),
    )

    # Using dictConfig to override existing loggers, which prevents failures in
    # test_main.py due to capsys not being cleared.
    logging.config.dictConfig(
        {
            "disable_existing_loggers": False,
            "version": 1,
            "handlers": {
                "console": {
                    "class": "rich.logging.RichHandler",
                    "show_time": False,
                    "show_path": False,
                    "highlighter": rich.highlighter.NullHighlighter(),
                }
            },
            "root": {
                "handlers": ["console"],
            },
        }
    )


def list_dependencies_and_versions() -> List[Tuple[str, str]]:
    deps = [
        "keyring",  # optional for non-desktop use
        "packaging",
        "requests",
        "requests-toolbelt",
        "urllib3",
        "id",
    ]
    if sys.version_info < (3, 10):
        deps.append("importlib-metadata")

    result: List[Tuple[str, str]] = []
    for dep in deps:
        try:
            version = importlib_metadata.version(dep)
        except importlib_metadata.PackageNotFoundError:
            version = "NOT INSTALLED"
        result.append((dep, version))

    return result


def dep_versions() -> str:
    return ", ".join(
        "{}: {}".format(*dependency) for dependency in list_dependencies_and_versions()
    )


def dispatch(argv: List[str]) -> Any:
    registered_commands = importlib_metadata.entry_points(
        group="twine.registered_commands"
    )

    parser = argparse.ArgumentParser(prog="twine")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s version {twine.__version__} ({dep_versions()})",
    )
    parser.add_argument(
        "--no-color",
        default=False,
        required=False,
        action="store_true",
        help="disable colored output",
    )
    parser.add_argument(
        "command",
        choices=registered_commands.names,
    )
    parser.add_argument(
        "args",
        help=argparse.SUPPRESS,
        nargs=argparse.REMAINDER,
    )
    parser.parse_args(argv, namespace=args)

    configure_output()

    main = registered_commands[args.command].load()

    return main(args.args)
