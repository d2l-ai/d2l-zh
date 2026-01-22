#!/usr/bin/env python3
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
import http
import logging
import sys
from typing import Any, cast

import requests

from twine import cli
from twine import exceptions

logger = logging.getLogger(__name__)


def main() -> Any:
    # Ensure that all errors are logged, even before argparse
    cli.configure_output()

    try:
        error = cli.dispatch(sys.argv[1:])
    except requests.HTTPError as exc:
        # Assuming this response will never be None
        response = cast(requests.Response, exc.response)

        error = True
        status_code = response.status_code
        status_phrase = http.HTTPStatus(status_code).phrase
        logger.error(
            f"{exc.__class__.__name__}: {status_code} {status_phrase} "
            f"from {response.url}\n"
            f"{response.reason}"
        )
    except exceptions.TwineException as exc:
        error = True
        logger.error(f"{exc.__class__.__name__}: {exc.args[0]}")

    return error


if __name__ == "__main__":
    sys.exit(main())
