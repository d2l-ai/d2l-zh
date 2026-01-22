# Copyright 2022 The Sigstore Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Ambient OIDC credential detection.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess  # nosec B404

import requests

from ... import AmbientCredentialError, GitHubOidcPermissionCredentialError

logger = logging.getLogger(__name__)

_GCP_PRODUCT_NAME_FILE = "/sys/class/dmi/id/product_name"
_GCP_TOKEN_REQUEST_URL = (
    "http://metadata/computeMetadata/v1/instance/service-accounts/default/token"  # noqa # nosec B105
)
_GCP_IDENTITY_REQUEST_URL = (
    "http://metadata/computeMetadata/v1/instance/service-accounts/default/identity"  # noqa
)
_GCP_GENERATEIDTOKEN_REQUEST_URL = (
    "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{}:generateIdToken"  # noqa
)

_env_var_regex = re.compile(r"[^A-Z0-9_]|^[^A-Z_]")


def detect_github(audience: str) -> str | None:
    """
    Detect and return a GitHub Actions ambient OIDC credential.

    Returns `None` if the context is not a GitHub Actions environment.

    Raises if the environment is GitHub Actions, but is incorrect or
    insufficiently permissioned for an OIDC credential.
    """

    logger.debug("GitHub: looking for OIDC credentials")
    if not os.getenv("GITHUB_ACTIONS"):
        logger.debug("GitHub: environment doesn't look like a GH action; giving up")
        return None

    # If we're running on a GitHub Action, we need to issue a GET request
    # to a special URL with a special bearer token. Both are stored in
    # the environment and are only present if the workflow has sufficient permissions.
    req_token = os.getenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    if not req_token:
        raise GitHubOidcPermissionCredentialError(
            "GitHub: missing or insufficient OIDC token permissions, the "
            "ACTIONS_ID_TOKEN_REQUEST_TOKEN environment variable was unset"
        )
    req_url = os.getenv("ACTIONS_ID_TOKEN_REQUEST_URL")
    if not req_url:
        raise GitHubOidcPermissionCredentialError(
            "GitHub: missing or insufficient OIDC token permissions, the "
            "ACTIONS_ID_TOKEN_REQUEST_URL environment variable was unset"
        )

    logger.debug("GitHub: requesting OIDC token")
    resp = requests.get(
        req_url,
        params={"audience": audience},
        headers={"Authorization": f"bearer {req_token}"},
        timeout=30,
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as http_error:
        raise AmbientCredentialError(
            f"GitHub: OIDC token request failed (code={resp.status_code}, "
            f"body={resp.content.decode()!r})"
        ) from http_error
    except requests.Timeout:
        raise AmbientCredentialError("GitHub: OIDC token request timed out")

    try:
        body = resp.json()
        value = body["value"]

        if not isinstance(value, str):
            raise ValueError("OIDC token is not a string")
    except Exception as e:
        raise AmbientCredentialError("GitHub: malformed or incomplete JSON") from e

    logger.debug("GitHub: successfully requested OIDC token")
    return value


def detect_gcp(audience: str) -> str | None:
    """
    Detect an return a Google Cloud Platform ambient OIDC credential.

    Returns `None` if the context is not a GCP environment.

    Raises if the environment is GCP, but is incorrect or
    insufficiently permissioned for an OIDC credential.
    """
    logger.debug("GCP: looking for OIDC credentials")

    service_account_name = os.getenv("GOOGLE_SERVICE_ACCOUNT_NAME")
    if service_account_name:
        logger.debug("GCP: GOOGLE_SERVICE_ACCOUNT_NAME set; attempting impersonation")

        logger.debug("GCP: requesting access token")
        resp = requests.get(
            _GCP_TOKEN_REQUEST_URL,
            params={"scopes": "https://www.googleapis.com/auth/cloud-platform"},
            headers={"Metadata-Flavor": "Google"},
            timeout=30,
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError as http_error:
            raise AmbientCredentialError(
                f"GCP: access token request failed (code={resp.status_code}, "
                f"body={resp.content.decode()!r})"
            ) from http_error
        except requests.Timeout:
            raise AmbientCredentialError("GCP: access token request timed out")

        access_token = resp.json().get("access_token")

        if not access_token:
            raise AmbientCredentialError("GCP: access token missing from response")

        resp = requests.post(
            _GCP_GENERATEIDTOKEN_REQUEST_URL.format(service_account_name),
            json={"audience": audience, "includeEmail": True},
            headers={
                "Authorization": f"Bearer {access_token}",
            },
            timeout=30,
        )

        logger.debug("GCP: requesting OIDC token")
        try:
            resp.raise_for_status()
        except requests.HTTPError as http_error:
            raise AmbientCredentialError(
                f"GCP: OIDC token request failed (code={resp.status_code}, "
                f"body={resp.content.decode()!r})"
            ) from http_error
        except requests.Timeout:
            raise AmbientCredentialError("GCP: OIDC token request timed out")

        oidc_token: str = resp.json().get("token")

        if not oidc_token:
            raise AmbientCredentialError("GCP: OIDC token missing from response")

        logger.debug("GCP: successfully requested OIDC token")
        return oidc_token

    else:
        logger.debug("GCP: GOOGLE_SERVICE_ACCOUNT_NAME not set; skipping impersonation")

        try:
            with open(_GCP_PRODUCT_NAME_FILE) as f:
                name = f.read().strip()
        except OSError:
            logger.debug("GCP: environment doesn't have GCP product name file; giving up")
            return None

        if name not in {"Google", "Google Compute Engine"}:
            logger.debug(f"GCP: product name file exists, but product name is {name!r}; giving up")
            return None

        logger.debug("GCP: requesting OIDC token")
        resp = requests.get(
            _GCP_IDENTITY_REQUEST_URL,
            params={"audience": audience, "format": "full"},
            headers={"Metadata-Flavor": "Google"},
            timeout=30,
        )

        try:
            resp.raise_for_status()
        except requests.HTTPError as http_error:
            raise AmbientCredentialError(
                f"GCP: OIDC token request failed (code={resp.status_code}, "
                f"body={resp.content.decode()!r})"
            ) from http_error
        except requests.Timeout:
            raise AmbientCredentialError("GCP: OIDC token request timed out")

        logger.debug("GCP: successfully requested OIDC token")
        return resp.text


def detect_buildkite(audience: str) -> str | None:
    """
    Detect and return a Buildkite ambient OIDC credential.

    Returns `None` if the context is not a Buildkite environment.

    Raises if the environment is Buildkite, but no Buildkite agent is found or
    the agent encounters an error when generating an OIDC token.
    """
    logger.debug("Buildkite: looking for OIDC credentials")

    if not os.getenv("BUILDKITE"):
        logger.debug("Buildkite: environment doesn't look like BuildKite; giving up")
        return None

    # Check that the Buildkite agent executable exists in the `PATH`.
    if shutil.which("buildkite-agent") is None:
        raise AmbientCredentialError(
            "Buildkite: could not find Buildkite agent in Buildkite environment"
        )

    # Now query the agent for a token.
    #
    # NOTE(alex): We're silencing `bandit` here. The reasoning for ignoring each
    # test are as follows.
    #
    # B603: This is complaining about invoking an external executable. However,
    # there doesn't seem to be any way to do this that satisfies `bandit` so I
    # think we need to ignore this.
    # More context at:
    #   https://github.com/PyCQA/bandit/issues/333
    #
    # B607: This is complaining about invoking an external executable without
    # providing an absolute path (we just refer to whatever `buildkite-agent`)
    # is in the `PATH`. For a Buildkite agent, there's no guarantee where the
    # `buildkite-agent` is installed so again, I don't think there's anything
    # we can do about this.
    process = subprocess.run(  # nosec B603, B607
        ["buildkite-agent", "oidc", "request-token", "--audience", audience],
        capture_output=True,
        text=True,
    )

    if process.returncode != 0:
        raise AmbientCredentialError(
            f"Buildkite: the Buildkite agent encountered an error: {process.stdout}"
        )

    return process.stdout.strip()


def detect_gitlab(audience: str) -> str | None:
    """
    Detect and return a GitLab CI/CD ambient OIDC credential.

    This detection is based on an environment variable. The variable name must be
    `<AUD>_ID_TOKEN`  where `<AUD>` is the uppercased audience argument where all
    characters outside of ASCII letters and digits are replaced with "_". A
    leading digit must also replaced with a "_".

    As an example, audience "sigstore" would require variable SIGSTORE_ID_TOKEN,
    and audience "http://test.audience" would require variable
    HTTP___TEST_AUDIENCE_ID_TOKEN.

    Returns `None` if the context is not GitLab CI/CD environment.

    Raises if the environment is GitLab, but the `<AUD>_ID_TOKEN` environment
    variable is not set.
    """
    logger.debug("GitLab: looking for OIDC credentials")

    if not os.getenv("GITLAB_CI"):
        logger.debug("GitLab: environment doesn't look like GitLab CI/CD; giving up")
        return None

    # construct a reasonable env var name from the audience
    sanitized_audience = _env_var_regex.sub("_", audience.upper())
    var_name = f"{sanitized_audience}_ID_TOKEN"
    token = os.getenv(var_name)
    if not token:
        raise AmbientCredentialError(f"GitLab: Environment variable {var_name} not found")

    logger.debug(f"GitLab: Found token in environment variable {var_name}")
    return token


def detect_circleci(audience: str) -> str | None:
    """
    Detect and return a CircleCI ambient OIDC credential.

    Returns `None` if the context is not a CircleCI environment.

    Raises if the environment is GitHub Actions, but is incorrect or
    insufficiently permissioned for an OIDC credential.
    """
    logger.debug("CircleCI: looking for OIDC credentials")

    if not os.getenv("CIRCLECI"):
        logger.debug("CircleCI: environment doesn't look like CircleCI; giving up")
        return None

    # Check that the circleci executable exists in the `PATH`.
    if shutil.which("circleci") is None:
        raise AmbientCredentialError("CircleCI: could not find `circleci` in the environment")

    # See NOTE on `detect_buildkite` for why we silence these warnings.
    payload = json.dumps({"aud": audience})
    process = subprocess.run(  # nosec B603, B607
        ["circleci", "run", "oidc", "get", "--claims", payload],
        capture_output=True,
        text=True,
    )

    if process.returncode != 0:
        raise AmbientCredentialError(
            f"CircleCI: the `circleci` tool encountered an error: {process.stdout}"
        )

    return process.stdout.strip()
