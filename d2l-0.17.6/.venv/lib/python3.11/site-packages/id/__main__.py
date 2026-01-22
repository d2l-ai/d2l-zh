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
The `python -m id` entrypoint.
"""

import argparse
import logging
import os

from . import __version__

logging.basicConfig()
logger = logging.getLogger(__name__)

# NOTE: We configure the top package logger, rather than the root logger,
# to avoid overly verbose logging in third-party code by default.
package_logger = logging.getLogger("id")
package_logger.setLevel(os.environ.get("ID_LOGLEVEL", "INFO").upper())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="id",
        description="a tool for generating OIDC identities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="run with additional debug logging; supply multiple times to increase verbosity",
    )
    parser.add_argument(
        "-d",
        "--decode",
        action="store_true",
        help="decode the OIDC token into JSON",
    )
    parser.add_argument(
        "audience",
        type=str,
        default=os.getenv("ID_OIDC_AUDIENCE"),
        help="the OIDC audience to use",
    )

    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()

    # Configure logging upfront, so that we don't miss anything.
    if args.verbose >= 1:
        package_logger.setLevel("DEBUG")
    if args.verbose >= 2:
        logging.getLogger().setLevel("DEBUG")

    logger.debug(f"parsed arguments {args}")

    from . import decode_oidc_token, detect_credential

    token = detect_credential(args.audience)
    if token and args.decode:
        header, payload, signature = decode_oidc_token(token)
        print(header)
        print(payload)
    else:
        print(token)


if __name__ == "__main__":  # pragma: no cover
    main()
