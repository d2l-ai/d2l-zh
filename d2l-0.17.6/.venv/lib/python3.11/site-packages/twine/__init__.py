"""Top-level module for Twine.

The contents of this package are not a public API. For more details, see
https://github.com/pypa/twine/issues/194 and https://github.com/pypa/twine/issues/665.
"""

# Copyright 2018 Donald Stufft and individual contributors
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
__all__ = (
    "__title__",
    "__summary__",
    "__uri__",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__copyright__",
)

__copyright__ = "Copyright 2019 Donald Stufft and individual contributors"

import email.utils
import sys

if sys.version_info >= (3, 10):
    import importlib.metadata as importlib_metadata
else:
    import importlib_metadata

metadata = importlib_metadata.metadata("twine")
assert metadata is not None  # nosec: B101


__title__ = metadata["name"]
__summary__ = metadata["summary"]
__uri__ = next(
    entry.split(", ")[1]
    for entry in metadata.get_all("Project-URL", ())
    if entry.startswith("Homepage")
)
__version__ = metadata["version"]
__author__, __email__ = email.utils.parseaddr(metadata["author-email"])
__license__ = None
