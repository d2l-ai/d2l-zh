import datetime
import functools
import getpass
import json
import logging
import time
import typing as t
from typing import cast
from urllib.parse import urlparse

import requests.auth
from id import AmbientCredentialError  # type: ignore
from id import detect_credential

# keyring has an indirect dependency on PyCA cryptography, which has no
# pre-built wheels for ppc64le and s390x, see #1158.
if t.TYPE_CHECKING:
    import keyring
    from keyring.errors import NoKeyringError
else:
    try:
        import keyring
        from keyring.errors import NoKeyringError
    except ModuleNotFoundError:  # pragma: no cover
        keyring = None
        NoKeyringError = None

from twine import exceptions
from twine import utils

logger = logging.getLogger(__name__)

TOKEN_USERNAME: t.Final[str] = "__token__"
#: Tokens expire after 15 minutes, let's start allowing renewal/replacement
#: after 10 minutes that way if we fail, we may still have time to replace it
#: before it expires. Thus, if our current time + this threshold is past the
#: greater or equal to the expiration time, we should start trying to replace
#: the token.
TOKEN_RENEWAL_THRESHOLD: t.Final[datetime.timedelta] = datetime.timedelta(
    minutes=5,
)


class CredentialInput:
    def __init__(
        self, username: t.Optional[str] = None, password: t.Optional[str] = None
    ) -> None:
        self.username = username
        self.password = password


class TrustedPublishingTokenRetrievalError(t.TypedDict):
    code: str
    description: str


class TrustedPublishingToken(t.TypedDict, total=False):
    message: t.Optional[str]
    errors: t.Optional[list[TrustedPublishingTokenRetrievalError]]
    token: t.Optional[str]
    success: t.Optional[bool]
    # Depends on https://github.com/pypi/warehouse/issues/18235
    expires: t.Optional[int]


class TrustedPublishingAuthenticator(requests.auth.AuthBase):
    def __init__(
        self,
        resolver: "Resolver",
    ) -> None:
        self.resolver = resolver

    def __call__(
        self, request: "requests.models.PreparedRequest"
    ) -> "requests.models.PreparedRequest":
        token = self.resolver.make_trusted_publishing_token()
        if token is None:
            raise exceptions.TrustedPublishingFailure(
                "Expected a trusted publishing token but got None"
            )

        # Instead of reconstructing Basic Auth headers ourself, let's just
        # rely on the underlying class to do the right thing.
        basic_auth = requests.auth.HTTPBasicAuth(
            username=TOKEN_USERNAME,
            password=token,
        )
        return cast(requests.models.PreparedRequest, basic_auth(request))


class Resolver:
    _tp_token: t.Optional[TrustedPublishingToken] = None
    _expires: t.Optional[int] = None

    def __init__(
        self,
        config: utils.RepositoryConfig,
        input: CredentialInput,
    ) -> None:
        self.config = config
        self.input = input

    @property
    @functools.lru_cache()
    def authenticator(self) -> "requests.auth.AuthBase":
        username = self.username
        password = self.password
        if self._tp_token:
            # If `self.password` ended up getting a Trusted Publishing token,
            # we've cached it here so we should use that as the authenticator.
            # We have a custom authenticator so we can repeatedly invoke
            # `make_trusted_publishing_token` which if the token is 10 minutes
            # old or more, we should get a new one automatically.
            return TrustedPublishingAuthenticator(resolver=self)
        if username and password:
            return requests.auth.HTTPBasicAuth(
                username=username,
                password=password,
            )
        raise exceptions.InvalidConfiguration(
            "could not determine credentials for configured repository"
        )

    @classmethod
    def choose(cls, interactive: bool) -> t.Type["Resolver"]:
        return cls if interactive else Private

    @property
    @functools.lru_cache()
    def username(self) -> t.Optional[str]:
        if self.is_pypi() and not self.input.username:
            # Default username.
            self.input.username = TOKEN_USERNAME

        return utils.get_userpass_value(
            self.input.username,
            self.config,
            key="username",
            prompt_strategy=self.username_from_keyring_or_prompt,
        )

    @property
    @functools.lru_cache()
    def password(self) -> t.Optional[str]:
        return utils.get_userpass_value(
            self.input.password,
            self.config,
            key="password",
            prompt_strategy=self.password_from_keyring_or_trusted_publishing_or_prompt,
        )

    def _has_valid_cached_tp_token(self) -> bool:
        return self._tp_token is not None and (
            int(time.time()) + TOKEN_RENEWAL_THRESHOLD.seconds
            < cast(int, self._tp_token.get("expires", self._expires))
        )

    def _make_trusted_publishing_token(self) -> t.Optional[TrustedPublishingToken]:
        if self._has_valid_cached_tp_token():
            return self._tp_token
        # Trusted publishing (OpenID Connect): get one token from the CI
        # system, and exchange that for a PyPI token.
        repository_domain = cast(str, urlparse(self.system).netloc)
        session = utils.make_requests_session()

        # Indices are expected to support `https://{domain}/_/oidc/audience`,
        # which tells OIDC exchange clients which audience to use.
        audience_url = f"https://{repository_domain}/_/oidc/audience"
        resp = session.get(audience_url, timeout=5)
        resp.raise_for_status()
        audience = cast(str, resp.json()["audience"])

        try:
            oidc_token = detect_credential(audience)
        except AmbientCredentialError as e:
            # If we get here, we're on a supported CI platform for trusted
            # publishing, and we have not been given any token, so we can error.
            raise exceptions.TrustedPublishingFailure(
                "Unable to retrieve an OIDC token from the CI platform for "
                f"trusted publishing {e}"
            )

        if oidc_token is None:
            logger.warning("This environment is not supported for trusted publishing")
            if self._tp_token and int(time.time()) > cast(
                int, self._tp_token.get("expires", self._expires)
            ):
                return None  # Fall back to prompting for a token (if possible)
            # The cached trusted publishing token may still be valid for a
            # while longer, let's continue using it instead of prompting
            return self._tp_token

        logger.warning("Got OIDC token for audience %s", audience)

        token_exchange_url = f"https://{repository_domain}/_/oidc/mint-token"

        mint_token_resp = session.post(
            token_exchange_url,
            json={"token": oidc_token},
            timeout=5,  # S113 wants a timeout
        )
        try:
            mint_token_payload = mint_token_resp.json()
        except json.JSONDecodeError:
            raise exceptions.TrustedPublishingFailure(
                "The token-minting request returned invalid JSON"
            )

        if not mint_token_resp.ok:
            reasons = "\n".join(
                f'* `{error["code"]}`: {error["description"]}'
                for error in mint_token_payload["errors"]
            )
            raise exceptions.TrustedPublishingFailure(
                "The token request failed; the index server gave the following"
                f" reasons:\n\n{reasons}"
            )

        logger.warning("Minted upload token for trusted publishing")
        self._tp_token = cast(TrustedPublishingToken, mint_token_payload)
        self._expires = int(time.time()) + 900
        return self._tp_token

    def make_trusted_publishing_token(self) -> t.Optional[str]:
        mint_token_payload = self._make_trusted_publishing_token()
        if not mint_token_payload:
            return None
        return cast(str, mint_token_payload["token"])

    @property
    def system(self) -> t.Optional[str]:
        return self.config["repository"]

    def get_username_from_keyring(self) -> t.Optional[str]:
        if keyring is None:
            logger.info("keyring module is not available")
            return None
        try:
            system = cast(str, self.system)
            logger.info("Querying keyring for username")
            creds = keyring.get_credential(system, None)
            if creds:
                return creds.username
        except AttributeError:
            # To support keyring prior to 15.2
            pass
        except Exception as exc:
            logger.warning("Error getting username from keyring", exc_info=exc)
        return None

    def get_password_from_keyring(self) -> t.Optional[str]:
        if keyring is None:
            logger.info("keyring module is not available")
            return None
        try:
            system = cast(str, self.system)
            username = cast(str, self.username)
            logger.info("Querying keyring for password")
            return cast(str, keyring.get_password(system, username))
        except NoKeyringError:
            logger.info("No keyring backend found")
        except Exception as exc:
            logger.warning("Error getting password from keyring", exc_info=exc)
        return None

    def username_from_keyring_or_prompt(self) -> str:
        username = self.get_username_from_keyring()
        if username:
            logger.info("username set from keyring")
            return username

        return self.prompt("username", input)

    def password_from_keyring_or_trusted_publishing_or_prompt(self) -> str:
        password = self.get_password_from_keyring()
        if password:
            logger.info("password set from keyring")
            return password

        if self.is_pypi() and self.username == TOKEN_USERNAME:
            logger.info(
                "Trying to use trusted publishing (no token was explicitly provided)"
            )
            if (token := self.make_trusted_publishing_token()) is not None:
                return token

        # Prompt for API token when required.
        what = "API token" if self.is_pypi() else "password"

        return self.prompt(what, getpass.getpass)

    def prompt(self, what: str, how: t.Callable[..., str]) -> str:
        return how(f"Enter your {what}: ")

    def is_pypi(self) -> bool:
        """As of 2024-01-01, PyPI requires API tokens for uploads."""
        return cast(str, self.config["repository"]).startswith(
            (
                utils.DEFAULT_REPOSITORY,
                utils.TEST_REPOSITORY,
            )
        )


class Private(Resolver):
    def prompt(self, what: str, how: t.Optional[t.Callable[..., str]] = None) -> str:
        raise exceptions.NonInteractive(f"Credential not found for {what}.")
