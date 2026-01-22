"""Module containing exceptions raised by twine."""

# Copyright 2015 Ian Stapleton Cordasco
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
import typing as t


class TwineException(Exception):
    """Base class for all exceptions raised by twine."""

    pass


class RedirectDetected(TwineException):
    """A redirect was detected that the user needs to resolve.

    In some cases, requests refuses to issue a new POST request after a
    redirect. In order to prevent a confusing user experience, we raise this
    exception to allow users to know the index they're uploading to is
    redirecting them.
    """

    @classmethod
    def from_args(cls, repository_url: str, redirect_url: str) -> "RedirectDetected":
        if redirect_url == f"{repository_url}/":
            return cls(
                f"{repository_url} attempted to redirect to {redirect_url}.\n"
                f"Your repository URL is missing a trailing slash. "
                "Please add it and try again.",
            )

        return cls(
            f"{repository_url} attempted to redirect to {redirect_url}.\n"
            f"If you trust these URLs, set {redirect_url} as your repository URL "
            "and try again.",
        )


class PackageNotFound(TwineException):
    """A package file was provided that could not be found on the file system.

    This is only used when attempting to register a package_file.
    """

    pass


class UploadToDeprecatedPyPIDetected(TwineException):
    """An upload attempt was detected to deprecated PyPI domains.

    The sites pypi.python.org and testpypi.python.org are deprecated.
    """

    @classmethod
    def from_args(
        cls, target_url: str, default_url: str, test_url: str
    ) -> "UploadToDeprecatedPyPIDetected":
        """Return an UploadToDeprecatedPyPIDetected instance."""
        return cls(
            "You're trying to upload to the legacy PyPI site '{}'. "
            "Uploading to those sites is deprecated. \n "
            "The new sites are pypi.org and test.pypi.org. Try using "
            "{} (or {}) to upload your packages instead. "
            "These are the default URLs for Twine now. \n More at "
            "https://packaging.python.org/guides/migrating-to-pypi-org/"
            " .".format(target_url, default_url, test_url)
        )


class UnsupportedConfiguration(TwineException):
    """An upload attempt was detected using features not supported by a repository.

    The features specified either in configuration or on the command-line.
    """

    class Builder:
        """Build the parameters for an UnsupportedConfiguration exception.

        In the event we add additional features we are not allowing with
        something other than PyPI or TestPyPI, we can use a builder to
        accumulate them all instead of requiring someone to run multiple times
        to discover all unsupported configuration options.
        """

        repository_url: str
        features: t.List[str]

        def __init__(self) -> None:
            self.repository_url = ""
            self.features = []

        def with_repository_url(
            self, repository_url: str
        ) -> "UnsupportedConfiguration.Builder":
            self.repository_url = repository_url
            return self

        def with_feature(self, feature: str) -> "UnsupportedConfiguration.Builder":
            self.features.append(feature)
            return self

        def finalize(self) -> "UnsupportedConfiguration":
            return UnsupportedConfiguration(
                f"The configured repository {self.repository_url!r} does not "
                "have support for the following features: "
                f"{', '.join(self.features)} and is an unsupported "
                "configuration",
                self.repository_url,
                *self.features,
            )


class UnreachableRepositoryURLDetected(TwineException):
    """An upload attempt was detected to a URL without a protocol prefix.

    All repository URLs must have a protocol (e.g., ``https://``).
    """

    pass


class InvalidSigningConfiguration(TwineException):
    """Both the sign and identity parameters must be present."""

    pass


class InvalidSigningExecutable(TwineException):
    """Signing executable must be installed on system."""

    pass


class InvalidConfiguration(TwineException):
    """Raised when configuration is invalid."""

    pass


class InvalidDistribution(TwineException):
    """Raised when a distribution is invalid."""

    pass


class NonInteractive(TwineException):
    """Raised in non-interactive mode when credentials could not be found."""

    pass


class TrustedPublishingFailure(TwineException):
    """Raised if we expected to use trusted publishing but couldn't."""

    pass


class InvalidPyPIUploadURL(TwineException):
    """Repository configuration tries to use PyPI with an incorrect URL.

    For example, https://pypi.org instead of https://upload.pypi.org/legacy.
    """

    pass
