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
import collections
import configparser
import functools
import logging
import os
import os.path
import unicodedata
from typing import Any, Callable, DefaultDict, Dict, Optional, Sequence, Union, cast
from urllib.parse import urlparse
from urllib.parse import urlunparse

import requests
import rfc3986
import urllib3
from requests.adapters import HTTPAdapter
from requests_toolbelt.utils import user_agent

import twine
from twine import exceptions

# Shim for input to allow testing.
input_func = input

DEFAULT_REPOSITORY = "https://upload.pypi.org/legacy/"
TEST_REPOSITORY = "https://test.pypi.org/legacy/"

DEFAULT_CONFIG_FILE = "~/.pypirc"

# TODO: In general, it seems to be assumed that the values retrieved from
# instances of this type aren't None, except for username and password.
# Type annotations would be cleaner if this were Dict[str, str], but that
# requires reworking the username/password handling, probably starting with
# get_userpass_value.
RepositoryConfig = Dict[str, Optional[str]]

logger = logging.getLogger(__name__)


def get_config(path: str) -> Dict[str, RepositoryConfig]:
    """Read repository configuration from a file (i.e. ~/.pypirc).

    Format: https://packaging.python.org/specifications/pypirc/

    If the default config file doesn't exist, return a default configuration for
    pypyi and testpypi.
    """
    realpath = os.path.realpath(os.path.expanduser(path))
    parser = configparser.RawConfigParser()

    try:
        with open(realpath) as f:
            parser.read_file(f)
            logger.info(f"Using configuration from {realpath}")
    except FileNotFoundError:
        # User probably set --config-file, but the file can't be read
        if path != DEFAULT_CONFIG_FILE:
            raise

    # server-login is obsolete, but retained for backwards compatibility
    defaults: RepositoryConfig = {
        "username": parser.get("server-login", "username", fallback=None),
        "password": parser.get("server-login", "password", fallback=None),
    }

    config: DefaultDict[str, RepositoryConfig]
    config = collections.defaultdict(lambda: defaults.copy())

    index_servers = parser.get(
        "distutils", "index-servers", fallback="pypi testpypi"
    ).split()

    # Don't require users to manually configure URLs for these repositories
    config["pypi"]["repository"] = DEFAULT_REPOSITORY
    if "testpypi" in index_servers:
        config["testpypi"]["repository"] = TEST_REPOSITORY

    # Optional configuration values for individual repositories
    for repository in index_servers:
        for key in [
            "username",
            "repository",
            "password",
            "ca_cert",
            "client_cert",
        ]:
            if parser.has_option(repository, key):
                config[repository][key] = parser.get(repository, key)

    # Convert the defaultdict to a regular dict to prevent surprising behavior later on
    return dict(config)


def sanitize_url(url: str) -> str:
    """Sanitize a URL.

    Sanitize URLs, removing any user:password combinations and replacing them with
    asterisks.  Returns the original URL if the string is a non-matching pattern.

    :param url:
        str containing a URL to sanitize.

    return:
        str either sanitized or as entered depending on pattern match.
    """
    uri = rfc3986.urlparse(url)
    if uri.userinfo:
        return cast(str, uri.copy_with(userinfo="*" * 8).unsplit())
    return url


def _validate_repository_url(repository_url: str) -> None:
    """Validate the given url for allowed schemes and components."""
    # Allowed schemes are http and https, based on whether the repository
    # supports TLS or not, and scheme and host must be present in the URL
    validator = (
        rfc3986.validators.Validator()
        .allow_schemes("http", "https")
        .require_presence_of("scheme", "host")
    )
    try:
        validator.validate(rfc3986.uri_reference(repository_url))
    except rfc3986.exceptions.RFC3986Exception as exc:
        raise exceptions.UnreachableRepositoryURLDetected(
            f"Invalid repository URL: {exc.args[0]}."
        )


def get_repository_from_config(
    config_file: str,
    repository: str,
    repository_url: Optional[str] = None,
) -> RepositoryConfig:
    """Get repository config command-line values or the .pypirc file."""
    # Prefer CLI `repository_url` over `repository` or .pypirc
    if repository_url:
        _validate_repository_url(repository_url)
        return _config_from_repository_url(repository_url)

    try:
        config = get_config(config_file)[repository]
    except OSError as exc:
        raise exceptions.InvalidConfiguration(str(exc))
    except KeyError:
        raise exceptions.InvalidConfiguration(
            f"Missing '{repository}' section from {config_file}.\n"
            f"More info: https://packaging.python.org/specifications/pypirc/ "
        )
    except configparser.Error:
        # NOTE: We intentionally fully mask the configparser exception here,
        # since it could leak tokens and other sensitive values.
        raise exceptions.InvalidConfiguration(
            f"Malformed configuration in {config_file}.\n"
            f"More info: https://packaging.python.org/specifications/pypirc/ "
        )

    config["repository"] = normalize_repository_url(cast(str, config["repository"]))
    return config


_HOSTNAMES = {
    "pypi.python.org",
    "testpypi.python.org",
    "upload.pypi.org",
    "test.pypi.org",
}


def _config_from_repository_url(url: str) -> RepositoryConfig:
    parsed = urlparse(url)
    config = {"repository": url, "username": None, "password": None}
    if parsed.username:
        config["username"] = parsed.username
        config["password"] = parsed.password
        config["repository"] = cast(
            str, rfc3986.urlparse(url).copy_with(userinfo=None).unsplit()
        )
    config["repository"] = normalize_repository_url(cast(str, config["repository"]))
    return config


def normalize_repository_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc in _HOSTNAMES:
        return urlunparse(("https",) + parsed[1:])
    return urlunparse(parsed)


def get_file_size(filename: str) -> str:
    """Return the size of a file in KB, or MB if >= 1024 KB."""
    file_size = os.path.getsize(filename) / 1024
    size_unit = "KB"

    if file_size > 1024:
        file_size = file_size / 1024
        size_unit = "MB"

    return f"{file_size:.1f} {size_unit}"


def check_status_code(response: requests.Response, verbose: bool) -> None:
    """Generate a helpful message based on the response from the repository.

    Raise a custom exception for recognized errors. Otherwise, print the
    response content (based on the verbose option) before re-raising the
    HTTPError.
    """
    if response.status_code == 410 and "pypi.python.org" in response.url:
        raise exceptions.UploadToDeprecatedPyPIDetected(
            f"It appears you're uploading to pypi.python.org (or "
            f"testpypi.python.org). You've received a 410 error response. "
            f"Uploading to those sites is deprecated. The new sites are "
            f"pypi.org and test.pypi.org. Try using {DEFAULT_REPOSITORY} (or "
            f"{TEST_REPOSITORY}) to upload your packages instead. These are "
            f"the default URLs for Twine now. More at "
            f"https://packaging.python.org/guides/migrating-to-pypi-org/."
        )
    elif response.status_code == 405 and "pypi.org" in response.url:
        raise exceptions.InvalidPyPIUploadURL(
            f"It appears you're trying to upload to pypi.org but have an "
            f"invalid URL. You probably want one of these two URLs: "
            f"{DEFAULT_REPOSITORY} or {TEST_REPOSITORY}. Check your "
            f"--repository-url value."
        )

    try:
        response.raise_for_status()
    except requests.HTTPError as err:
        if not verbose:
            logger.warning(
                "Error during upload. "
                "Retry with the --verbose option for more details."
            )

        raise err


def get_userpass_value(
    cli_value: Optional[str],
    config: RepositoryConfig,
    key: str,
    prompt_strategy: Optional[Callable[[], str]] = None,
) -> Optional[str]:
    """Get a credential (e.g. a username or password) from the configuration.

    Uses the following rules:

    1. If ``cli_value`` is specified, use that.
    2. If ``config[key]`` is specified, use that.
    3. If ``prompt_strategy`` is specified, use its return value.
    4. Otherwise return ``None``

    :param cli_value:
        The value supplied from the command line.
    :param config:
        A dictionary of repository configuration values.
    :param key:
        The credential to look up in ``config``, e.g. ``"username"`` or ``"password"``.
    :param prompt_strategy:
        An argumentless function to get the value, e.g. from keyring or by prompting
        the user.

    :return:
        The credential value, i.e. the username or password.
    """
    if cli_value is not None:
        logger.info(f"{key} set by command options")
        return cli_value

    elif config.get(key) is not None:
        logger.info(f"{key} set from config file")
        return config[key]

    elif prompt_strategy:
        warning = ""
        value = prompt_strategy()

        if not value:
            warning = f"Your {key} is empty"
        elif any(unicodedata.category(c).startswith("C") for c in value):
            # See https://www.unicode.org/reports/tr44/#General_Category_Values
            # Most common case is "\x16" when pasting in Windows Command Prompt
            warning = f"Your {key} contains control characters"

        if warning:
            logger.warning(f"{warning}. Did you enter it correctly?")
            logger.warning(
                "See https://twine.readthedocs.io/#entering-credentials "
                "for more information."
            )

        return value

    else:
        return None


#: Get the CA bundle via :func:`get_userpass_value`.
get_cacert = functools.partial(get_userpass_value, key="ca_cert")

#: Get the client certificate via :func:`get_userpass_value`.
get_clientcert = functools.partial(get_userpass_value, key="client_cert")


def make_requests_session() -> requests.Session:
    """Prepare a requests Session with retries & twine's user-agent string."""
    s = requests.Session()

    retry = urllib3.Retry(
        allowed_methods=["GET"],
        connect=5,
        total=10,
        status_forcelist=[500, 501, 502, 503],
    )

    for scheme in ("http://", "https://"):
        s.mount(scheme, HTTPAdapter(max_retries=retry))

    s.headers["User-Agent"] = (
        user_agent.UserAgentBuilder("twine", twine.__version__)
        .include_implementation()
        .build()
    )
    return s


class EnvironmentDefault(argparse.Action):
    """Get values from environment variable."""

    def __init__(
        self,
        env: str,
        required: bool = True,
        default: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        default = os.environ.get(env, default)
        self.env = env
        if default:
            required = False
        super().__init__(default=default, required=required, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Union[str, Sequence[Any], None],
        option_string: Optional[str] = None,
    ) -> None:
        setattr(namespace, self.dest, values)


class EnvironmentFlag(argparse.Action):
    """Set boolean flag from environment variable."""

    def __init__(self, env: str, **kwargs: Any) -> None:
        default = self.bool_from_env(os.environ.get(env))
        self.env = env
        super().__init__(default=default, nargs=0, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Union[str, Sequence[Any], None],
        option_string: Optional[str] = None,
    ) -> None:
        setattr(namespace, self.dest, True)

    @staticmethod
    def bool_from_env(val: Optional[str]) -> bool:
        """Allow '0' and 'false' and 'no' to be False."""
        falsey = {"0", "false", "no"}
        return bool(val and val.lower() not in falsey)
