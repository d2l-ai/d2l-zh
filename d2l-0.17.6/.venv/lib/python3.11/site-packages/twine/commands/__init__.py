"""Module containing the logic for the ``twine`` sub-commands.

The contents of this package are not a public API. For more details, see
https://github.com/pypa/twine/issues/194 and https://github.com/pypa/twine/issues/665.
"""

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
import fnmatch
import glob
import os.path
from typing import Dict, List, NamedTuple

from twine import exceptions

__all__: List[str] = []


def _group_wheel_files_first(files: List[str]) -> List[str]:
    if not any(fname for fname in files if fname.endswith(".whl")):
        # Return early if there's no wheel files
        return files

    files.sort(key=lambda x: -1 if x.endswith(".whl") else 0)

    return files


def _find_dists(dists: List[str]) -> List[str]:
    uploads = []
    for filename in dists:
        if os.path.exists(filename):
            uploads.append(filename)
            continue
        # The filename didn't exist so it may be a glob
        files = glob.glob(filename)
        # If nothing matches, files is []
        if not files:
            raise exceptions.InvalidDistribution(
                "Cannot find file (or expand pattern): '%s'" % filename
            )
        # Otherwise, files will be filenames that exist
        uploads.extend(files)
    return _group_wheel_files_first(uploads)


class Inputs(NamedTuple):
    """Represents structured user inputs."""

    dists: List[str]
    signatures: Dict[str, str]
    attestations_by_dist: Dict[str, List[str]]


def _split_inputs(
    inputs: List[str],
) -> Inputs:
    """
    Split the unstructured list of input files provided by the user into groups.

    Three groups are returned: upload files (i.e. dists), signatures, and attestations.

    Upload files are returned as a linear list, signatures are returned as a
    dict of ``basename -> path``, and attestations are returned as a dict of
    ``dist-path -> [attestation-path]``.
    """
    signatures = {os.path.basename(i): i for i in fnmatch.filter(inputs, "*.asc")}
    attestations = fnmatch.filter(inputs, "*.*.attestation")
    dists = [
        dist
        for dist in inputs
        if dist not in (set(signatures.values()) | set(attestations))
    ]

    attestations_by_dist = {}
    for dist in dists:
        dist_basename = os.path.basename(dist)
        attestations_by_dist[dist] = [
            a for a in attestations if os.path.basename(a).startswith(dist_basename)
        ]

    return Inputs(dists, signatures, attestations_by_dist)
