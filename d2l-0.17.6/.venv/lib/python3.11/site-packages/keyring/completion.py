import argparse
import sys
from importlib.resources import files

try:
    import shtab
except ImportError:
    pass


class _MissingCompletionAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string):
        print("Install keyring[completion] for completion support.", file=sys.stderr)
        parser.exit(1)


def add_completion_notice(parser):
    """Add completion argument to parser."""
    parser.add_argument(
        "--print-completion",
        choices=["bash", "zsh", "tcsh"],
        action=_MissingCompletionAction,
        help="print shell completion script",
    )
    return parser


def get_action(parser, option):
    (match,) = (action for action in parser._actions if option in action.option_strings)
    return match


def install_completion(parser):
    preamble = dict(
        bash=files(__package__)
        .joinpath('backend_complete.bash')
        .read_text(encoding='utf-8'),
        zsh=files(__package__)
        .joinpath('backend_complete.zsh')
        .read_text(encoding='utf-8'),
    )
    shtab.add_argument_to(parser, preamble=preamble)
    get_action(parser, '--keyring-path').complete = shtab.DIR
    get_action(parser, '--keyring-backend').complete = dict(
        bash='_keyring_backends',
        zsh='backend_complete',
    )
    return parser


def install(parser):
    try:
        install_completion(parser)
    except NameError:
        add_completion_notice(parser)
