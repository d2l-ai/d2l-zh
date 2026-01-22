# Copyright 2015 Ian Cordasco
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
import hashlib
import io
import json
import logging
import os
import re
import subprocess
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, TypedDict

from packaging import metadata
from packaging import version
from rich import print

from twine import exceptions
from twine import sdist
from twine import wheel

# Monkeypatch Metadata 2.0 support
metadata._VALID_METADATA_VERSIONS = [
    "1.0",
    "1.1",
    "1.2",
    "2.0",
    "2.1",
    "2.2",
    "2.3",
    "2.4",
]

DIST_TYPES = {
    "bdist_wheel": wheel.Wheel,
    "sdist": sdist.SDist,
}

DIST_EXTENSIONS = {
    ".whl": "bdist_wheel",
    ".tar.gz": "sdist",
    ".zip": "sdist",
}

logger = logging.getLogger(__name__)


def _safe_name(name: str) -> str:
    """Convert an arbitrary string to a standard distribution name.

    Any runs of non-alphanumeric/. characters are replaced with a single '-'.

    Copied from pkg_resources.safe_name for compatibility with warehouse.
    See https://github.com/pypa/twine/issues/743.
    """
    return re.sub("[^A-Za-z0-9.]+", "-", name)


# Map ``metadata.RawMetadata`` fields to ``PackageMetadata`` fields.  Some
# fields are renamed to match the names expected in the upload form.
_RAW_TO_PACKAGE_METADATA = {
    # Metadata 1.0 - PEP 241
    "metadata_version": "metadata_version",
    "name": "name",
    "version": "version",
    "platforms": "platform",  # Renamed
    "summary": "summary",
    "description": "description",
    "keywords": "keywords",
    "home_page": "home_page",
    "author": "author",
    "author_email": "author_email",
    "license": "license",
    # Metadata 1.1 - PEP 314
    "supported_platforms": "supported_platform",  # Renamed
    "download_url": "download_url",
    "classifiers": "classifiers",
    "requires": "requires",
    "provides": "provides",
    "obsoletes": "obsoletes",
    # Metadata 1.2 - PEP 345
    "maintainer": "maintainer",
    "maintainer_email": "maintainer_email",
    "requires_dist": "requires_dist",
    "provides_dist": "provides_dist",
    "obsoletes_dist": "obsoletes_dist",
    "requires_python": "requires_python",
    "requires_external": "requires_external",
    "project_urls": "project_urls",
    # Metadata 2.1 - PEP 566
    "description_content_type": "description_content_type",
    "provides_extra": "provides_extra",
    # Metadata 2.2 - PEP 643
    "dynamic": "dynamic",
    # Metadata 2.4 - PEP 639
    "license_expression": "license_expression",
    "license_files": "license_file",  # Renamed
}


class PackageMetadata(TypedDict, total=False):

    # Metadata 1.0 - PEP 241
    metadata_version: str
    name: str
    version: str
    platform: List[str]
    summary: str
    description: str
    keywords: List[str]
    home_page: str
    author: str
    author_email: str
    license: str

    # Metadata 1.1 - PEP 314
    supported_platform: List[str]
    download_url: str
    classifiers: List[str]
    requires: List[str]
    provides: List[str]
    obsoletes: List[str]

    # Metadata 1.2 - PEP 345
    maintainer: str
    maintainer_email: str
    requires_dist: List[str]
    provides_dist: List[str]
    obsoletes_dist: List[str]
    requires_python: str
    requires_external: List[str]
    project_urls: Dict[str, str]

    # Metadata 2.1 - PEP 566
    description_content_type: str
    provides_extra: List[str]

    # Metadata 2.2 - PEP 643
    dynamic: List[str]

    # Metadata 2.4 - PEP 639
    license_expression: str
    license_file: List[str]

    # Additional metadata
    comment: str
    pyversion: str
    filetype: str
    gpg_signature: Tuple[str, bytes]
    attestations: str
    sha256_digest: str
    blake2_256_digest: str


class PackageFile:
    def __init__(
        self,
        filename: str,
        comment: Optional[str],
        metadata: metadata.RawMetadata,
        python_version: str,
        filetype: str,
    ) -> None:
        self.filename = filename
        self.basefilename = os.path.basename(filename)
        self.comment = comment
        self.metadata = metadata
        self.python_version = python_version
        self.filetype = filetype
        self.safe_name = _safe_name(metadata["name"])
        self.version: str = metadata["version"]
        self.signed_filename = self.filename + ".asc"
        self.signed_basefilename = self.basefilename + ".asc"
        self.gpg_signature: Optional[Tuple[str, bytes]] = None
        self.attestations: Optional[List[Dict[Any, str]]] = None

        hasher = HashManager(filename)
        hasher.hash()
        hexdigest = hasher.hexdigest()

        self.sha2_digest = hexdigest.sha2
        self.blake2_256_digest = hexdigest.blake2

    @classmethod
    def from_filename(cls, filename: str, comment: Optional[str]) -> "PackageFile":
        # Extract the metadata from the package
        for ext, dtype in DIST_EXTENSIONS.items():
            if filename.endswith(ext):
                try:
                    dist = DIST_TYPES[dtype](filename)
                    data = dist.read()
                    py_version = dist.py_version
                except EOFError:
                    raise exceptions.InvalidDistribution(
                        "Invalid distribution file: '%s'" % os.path.basename(filename)
                    )
                else:
                    break
        else:
            raise exceptions.InvalidDistribution(
                "Unknown distribution format: '%s'" % os.path.basename(filename)
            )

        # Parse and validate metadata.
        meta, unparsed = metadata.parse_email(data)

        # setuptools emits License-File metadata fields while declaring
        # Metadata-Version 2.1. This is invalid because the metadata
        # specification does not allow to add arbitrary fields, and because
        # the semantic implemented by setuptools is different than the one
        # described in PEP 639. However, rejecting these packages would be
        # too disruptive. Drop License-File metadata entries from the data
        # sent to the package index if the declared metadata version is less
        # than 2.4.
        if version.Version(meta.get("metadata_version", "0")) < version.Version("2.4"):
            meta.pop("license_files", None)
            # Support for metadata version 2.4 requires packaging version 24.1
            # or later. When parsing metadata with an older packaging, the
            # invalid License-File fields are not understood and added to the
            # unparsed dictionary. Remove them to avoid triggering the
            # following check.
            unparsed.pop("license-file", None)

        if unparsed:
            raise exceptions.InvalidDistribution(
                "Invalid distribution metadata: {}".format(
                    "; ".join(
                        f"unrecognized or malformed field {key!r}" for key in unparsed
                    )
                )
            )

        try:
            metadata.Metadata.from_raw(meta)
        except metadata.ExceptionGroup as group:
            raise exceptions.InvalidDistribution(
                "Invalid distribution metadata: {}".format(
                    "; ".join(sorted(str(e) for e in group.exceptions))
                )
            )

        return cls(filename, comment, meta, py_version, dtype)

    def metadata_dictionary(self) -> PackageMetadata:
        """Merge multiple sources of metadata into a single dictionary.

        Includes values from filename, PKG-INFO, hashers, and signature.
        """
        data = PackageMetadata()
        for key, value in self.metadata.items():
            field = _RAW_TO_PACKAGE_METADATA.get(key)
            if field:
                # A ``TypedDict`` only support literal key names. Here key
                # names are computed but they can only be valid key names.
                data[field] = value  # type: ignore[literal-required]

        # Override name with safe name.
        data["name"] = self.safe_name

        # File content.
        data["pyversion"] = self.python_version
        data["filetype"] = self.filetype

        # Additional meta-data: some of these fileds may not be set. Some
        # package repositories do not allow null values, so this only sends
        # non-null values. In particular, FIPS disables Blake2, making
        # the digest values null. See https://github.com/pypa/twine/issues/775

        if self.comment is not None:
            data["comment"] = self.comment

        if self.sha2_digest is not None:
            data["sha256_digest"] = self.sha2_digest

        if self.gpg_signature is not None:
            data["gpg_signature"] = self.gpg_signature

        if self.attestations is not None:
            data["attestations"] = json.dumps(self.attestations)

        if self.blake2_256_digest:
            data["blake2_256_digest"] = self.blake2_256_digest

        return data

    def add_attestations(self, attestations: List[str]) -> None:
        loaded_attestations = []
        for attestation in attestations:
            with open(attestation, "rb") as att:
                try:
                    loaded_attestations.append(json.load(att))
                except json.JSONDecodeError:
                    raise exceptions.InvalidDistribution(
                        f"invalid JSON in attestation: {attestation}"
                    )

        self.attestations = loaded_attestations

    def add_gpg_signature(
        self, signature_filepath: str, signature_filename: str
    ) -> None:
        if self.gpg_signature is not None:
            raise exceptions.InvalidDistribution("GPG Signature can only be added once")

        with open(signature_filepath, "rb") as gpg:
            self.gpg_signature = (signature_filename, gpg.read())

    def sign(self, sign_with: str, identity: Optional[str]) -> None:
        print(f"Signing {self.basefilename}")
        gpg_args: Tuple[str, ...] = (sign_with, "--detach-sign")
        if identity:
            gpg_args += ("--local-user", identity)
        gpg_args += ("-a", self.filename)
        self.run_gpg(gpg_args)

        self.add_gpg_signature(self.signed_filename, self.signed_basefilename)

    @classmethod
    def run_gpg(cls, gpg_args: Tuple[str, ...]) -> None:
        try:
            subprocess.check_call(gpg_args)
            return
        except FileNotFoundError:
            if gpg_args[0] != "gpg":
                raise exceptions.InvalidSigningExecutable(
                    f"{gpg_args[0]} executable not available."
                )

        logger.warning("gpg executable not available. Attempting fallback to gpg2.")
        try:
            subprocess.check_call(("gpg2",) + gpg_args[1:])
        except FileNotFoundError:
            raise exceptions.InvalidSigningExecutable(
                "'gpg' or 'gpg2' executables not available.\n"
                "Try installing one of these or specifying an executable "
                "with the --sign-with flag."
            )


class Hexdigest(NamedTuple):
    sha2: Optional[str]
    blake2: Optional[str]


class HashManager:
    """Manage our hashing objects for simplicity.

    This will also allow us to better test this logic.
    """

    def __init__(self, filename: str) -> None:
        """Initialize our manager and hasher objects."""
        self.filename = filename

        self._sha2_hasher = hashlib.sha256()

        self._blake_hasher = None
        try:
            self._blake_hasher = hashlib.blake2b(digest_size=256 // 8)
        except (ValueError, TypeError, AttributeError):
            # FIPS mode disables blake2
            pass

    def _sha2_update(self, content: bytes) -> None:
        if self._sha2_hasher is not None:
            self._sha2_hasher.update(content)

    def _sha2_hexdigest(self) -> Optional[str]:
        if self._sha2_hasher is not None:
            return self._sha2_hasher.hexdigest()
        return None

    def _blake_update(self, content: bytes) -> None:
        if self._blake_hasher is not None:
            self._blake_hasher.update(content)

    def _blake_hexdigest(self) -> Optional[str]:
        if self._blake_hasher is not None:
            return self._blake_hasher.hexdigest()
        return None

    def hash(self) -> None:
        """Hash the file contents."""
        with open(self.filename, "rb") as fp:
            for content in iter(lambda: fp.read(io.DEFAULT_BUFFER_SIZE), b""):
                self._sha2_update(content)
                self._blake_update(content)

    def hexdigest(self) -> Hexdigest:
        """Return the hexdigest for the file."""
        return Hexdigest(
            self._sha2_hexdigest(),
            self._blake_hexdigest(),
        )
