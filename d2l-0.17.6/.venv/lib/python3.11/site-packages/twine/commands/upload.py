"""Module containing the logic for ``twine upload``."""

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
import logging
from typing import Dict, List, cast

import requests
from rich import print

from twine import commands
from twine import exceptions
from twine import package as package_file
from twine import settings
from twine import utils

logger = logging.getLogger(__name__)


def skip_upload(
    response: requests.Response, skip_existing: bool, package: package_file.PackageFile
) -> bool:
    """Determine if a failed upload is an error or can be safely ignored.

    :param response:
        The response from attempting to upload ``package`` to a repository.
    :param skip_existing:
        If ``True``, use the status and content of ``response`` to determine if the
        package already exists on the repository. If so, then a failed upload is safe
        to ignore.
    :param package:
        The package that was being uploaded.

    :return:
        ``True`` if a failed upload can be safely ignored, otherwise ``False``.
    """
    if not skip_existing:
        return False

    status = response.status_code
    reason = getattr(response, "reason", "").lower()
    text = getattr(response, "text", "").lower()

    # NOTE(sigmavirus24): PyPI presently returns a 400 status code with the
    # error message in the reason attribute. Other implementations return a
    # 403 or 409 status code.
    return (
        # pypiserver (https://pypi.org/project/pypiserver)
        status == 409
        # PyPI / TestPyPI / GCP Artifact Registry
        or (status == 400 and any("already exist" in x for x in [reason, text]))
    )


def _make_package(
    filename: str,
    signatures: Dict[str, str],
    attestations: List[str],
    upload_settings: settings.Settings,
) -> package_file.PackageFile:
    """Create and sign a package, based off of filename, signatures, and settings.

    Additionally, any supplied attestations are attached to the package when
    the settings indicate to do so.
    """
    package = package_file.PackageFile.from_filename(filename, upload_settings.comment)

    signed_name = package.signed_basefilename
    if signed_name in signatures:
        package.add_gpg_signature(signatures[signed_name], signed_name)
    elif upload_settings.sign:
        package.sign(upload_settings.sign_with, upload_settings.identity)

    # Attestations are only attached if explicitly requested with `--attestations`.
    if upload_settings.attestations:
        # Passing `--attestations` without any actual attestations present
        # indicates user confusion, so we fail rather than silently allowing it.
        if not attestations:
            raise exceptions.InvalidDistribution(
                "Upload with attestations requested, but "
                f"{filename} has no associated attestations"
            )
        package.add_attestations(attestations)

    file_size = utils.get_file_size(package.filename)
    logger.info(f"{package.filename} ({file_size})")
    if package.gpg_signature:
        logger.info(f"Signed with {package.signed_filename}")

    return package


def upload(upload_settings: settings.Settings, dists: List[str]) -> None:
    """Upload one or more distributions to a repository, and display the progress.

    If a package already exists on the repository, most repositories will return an
    error response. However, if ``upload_settings.skip_existing`` is ``True``, a message
    will be displayed and any remaining distributions will be uploaded.

    For known repositories (like PyPI), the web URLs of successfully uploaded packages
    will be displayed.

    :param upload_settings:
        The configured options related to uploading to a repository.
    :param dists:
        The distribution files to upload to the repository. This can also include
        ``.asc`` and ``.attestation`` files, which will be added to their respective
        file uploads.

    :raises twine.exceptions.TwineException:
        The upload failed due to a configuration error.
    :raises requests.HTTPError:
        The repository responded with an error.
    """
    upload_settings.check_repository_url()
    upload_settings.verify_feature_capability()
    repository_url = cast(str, upload_settings.repository_config["repository"])

    # Attestations are only supported on PyPI and TestPyPI at the moment.
    # We warn instead of failing to allow twine to be used in local testing
    # setups (where the PyPI deployment doesn't have a well-known domain).
    if upload_settings.attestations and not repository_url.startswith(
        (utils.DEFAULT_REPOSITORY, utils.TEST_REPOSITORY)
    ):
        logger.warning(
            "Only PyPI and TestPyPI support attestations; "
            "if you experience failures, remove the --attestations flag and "
            "re-try this command"
        )

    dists = commands._find_dists(dists)
    # Determine if the user has passed in pre-signed distributions or any attestations.
    uploads, signatures, attestations_by_dist = commands._split_inputs(dists)

    print(f"Uploading distributions to {utils.sanitize_url(repository_url)}")

    packages_to_upload = [
        _make_package(
            filename, signatures, attestations_by_dist[filename], upload_settings
        )
        for filename in uploads
    ]

    if any(p.gpg_signature for p in packages_to_upload):
        if repository_url.startswith((utils.DEFAULT_REPOSITORY, utils.TEST_REPOSITORY)):
            # Warn the user if they're trying to upload a PGP signature to PyPI
            # or TestPyPI, which will (as of May 2023) ignore it.
            # This warning is currently limited to just those indices, since other
            # indices may still support PGP signatures.
            logger.warning(
                "One or more packages has an associated PGP signature; "
                "these will be silently ignored by the index"
            )
        else:
            # On other indices, warn the user that twine is considering
            # removing PGP support outright.
            logger.warning(
                "One or more packages has an associated PGP signature; "
                "a future version of twine may silently ignore these. "
                "See https://github.com/pypa/twine/issues/1009 for more "
                "information"
            )

    repository = upload_settings.create_repository()
    uploaded_packages = []

    if signatures and not packages_to_upload:
        raise exceptions.InvalidDistribution(
            "Cannot upload signed files by themselves, must upload with a "
            "corresponding distribution file."
        )

    for package in packages_to_upload:
        skip_message = (
            f"Skipping {package.basefilename} because it appears to already exist"
        )

        # Note: The skip_existing check *needs* to be first, because otherwise
        #       we're going to generate extra HTTP requests against a hardcoded
        #       URL for no reason.
        if upload_settings.skip_existing and repository.package_is_uploaded(package):
            logger.warning(skip_message)
            continue

        resp = repository.upload(package)
        logger.info(f"Response from {resp.url}:\n{resp.status_code} {resp.reason}")
        if resp.text:
            logger.info(resp.text)

        # Bug 92. If we get a redirect we should abort because something seems
        # funky. The behaviour is not well defined and redirects being issued
        # by PyPI should never happen in reality. This should catch malicious
        # redirects as well.
        if resp.is_redirect:
            raise exceptions.RedirectDetected.from_args(
                utils.sanitize_url(repository_url),
                utils.sanitize_url(resp.headers["location"]),
            )

        if skip_upload(resp, upload_settings.skip_existing, package):
            logger.warning(skip_message)
            continue

        utils.check_status_code(resp, upload_settings.verbose)

        uploaded_packages.append(package)

    release_urls = repository.release_urls(uploaded_packages)
    if release_urls:
        print("\n[green]View at:")
        for url in release_urls:
            print(url)

    # Bug 28. Try to silence a ResourceWarning by clearing the connection
    # pool.
    repository.close()


def main(args: List[str]) -> None:
    """Execute the ``upload`` command.

    :param args:
        The command-line arguments.
    """
    parser = argparse.ArgumentParser(prog="twine upload")
    settings.Settings.register_argparse_arguments(parser)
    parser.add_argument(
        "dists",
        nargs="+",
        metavar="dist",
        help="The distribution files to upload to the repository "
        "(package index). Usually dist/* . May additionally contain "
        "a .asc file to include an existing signature with the "
        "file upload.",
    )

    parsed_args = parser.parse_args(args)
    upload_settings = settings.Settings.from_argparse(parsed_args)

    # Call the upload function with the arguments from the command line
    return upload(upload_settings, parsed_args.dists)
