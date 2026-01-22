import os
import tarfile
import zipfile
from contextlib import suppress

from twine import distribution
from twine import exceptions


class SDist(distribution.Distribution):
    def __new__(cls, filename: str) -> "SDist":
        if cls is not SDist:
            return object.__new__(cls)

        FORMATS = {
            ".tar.gz": TarGzSDist,
            ".zip": ZipSDist,
        }

        for extension, impl in FORMATS.items():
            if filename.endswith(extension):
                return impl(filename)
        raise exceptions.InvalidDistribution(f"Unsupported sdist format: {filename}")

    def __init__(self, filename: str) -> None:
        if not os.path.exists(filename):
            raise exceptions.InvalidDistribution(f"No such file: {filename}")
        self.filename = filename

    @property
    def py_version(self) -> str:
        return "source"


class TarGzSDist(SDist):

    def read(self) -> bytes:
        with tarfile.open(self.filename, "r:gz") as sdist:
            # The sdist must contain a single top-level direcotry...
            root = os.path.commonpath(sdist.getnames())
            if root in {".", "/", ""}:
                raise exceptions.InvalidDistribution(
                    f"Too many top-level members in sdist archive: {self.filename}"
                )
            # ...containing the package metadata in a ``PKG-INFO`` file.
            with suppress(KeyError):
                member = sdist.getmember(root.rstrip("/") + "/PKG-INFO")
                if not member.isfile():
                    raise exceptions.InvalidDistribution(
                        f"PKG-INFO is not a regular file: {self.filename}"
                    )
                fd = sdist.extractfile(member)
                assert fd is not None, "for mypy"
                data = fd.read()
                if b"Metadata-Version" in data:
                    return data

        raise exceptions.InvalidDistribution(
            "No PKG-INFO in archive or "
            f"PKG-INFO missing 'Metadata-Version': {self.filename}"
        )


class ZipSDist(SDist):

    def read(self) -> bytes:
        with zipfile.ZipFile(self.filename) as sdist:
            # The sdist must contain a single top-level direcotry...
            root = os.path.commonpath(sdist.namelist())
            if root in {".", "/", ""}:
                raise exceptions.InvalidDistribution(
                    f"Too many top-level members in sdist archive: {self.filename}"
                )
            # ...containing the package metadata in a ``PKG-INFO`` file.
            with suppress(KeyError):
                data = sdist.read(root.rstrip("/") + "/PKG-INFO")
                if b"Metadata-Version" in data:
                    return data

        raise exceptions.InvalidDistribution(
            "No PKG-INFO in archive or "
            f"PKG-INFO missing 'Metadata-Version': {self.filename}"
        )
