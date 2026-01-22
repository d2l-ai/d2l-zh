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
API for retrieving OIDC tokens.
"""

from __future__ import annotations

import base64
from typing import Callable

__version__ = "1.5.0"


class IdentityError(Exception):
    """
    Raised on any OIDC token format or claim error.
    """

    pass


class AmbientCredentialError(IdentityError):
    """
    Raised when an ambient credential should be present, but
    can't be retrieved (e.g. network failure).
    """

    pass


class GitHubOidcPermissionCredentialError(AmbientCredentialError):
    """
    Raised when the current GitHub Actions environment doesn't have permission
    to retrieve an OIDC token.
    """

    pass


def detect_credential(audience: str) -> str | None:
    """
    Try each ambient credential detector, returning the first one to succeed
    or `None` if all fail.

    Raises `AmbientCredentialError` if any detector fails internally (i.e.
    detects a credential, but cannot retrieve it).
    """
    from ._internal.oidc.ambient import (
        detect_buildkite,
        detect_circleci,
        detect_gcp,
        detect_github,
        detect_gitlab,
    )

    detectors: list[Callable[..., str | None]] = [
        detect_github,
        detect_gcp,
        detect_buildkite,
        detect_gitlab,
        detect_circleci,
    ]
    for detector in detectors:
        credential = detector(audience)
        if credential is not None:
            return credential
    return None


def decode_oidc_token(token: str) -> tuple[str, str, str]:
    # Split the token into its three parts: header, payload, and signature
    header, payload, signature = token.split(".")

    # Decode base64-encoded header and payload
    decoded_header = base64.urlsafe_b64decode(header + "==").decode("utf-8")
    decoded_payload = base64.urlsafe_b64decode(payload + "==").decode("utf-8")

    return decoded_header, decoded_payload, signature
