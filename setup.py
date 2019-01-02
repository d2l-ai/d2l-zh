#!/usr/bin/env python
import io
import os
import re
from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version('d2lzh', '__init__.py')

requirements = [
    'numpy',
    'matplotlib',
	'jupyter'
]

setup(
    # Metadata
    name='d2lzh',
    version=VERSION,
    author='Contributors',
    author_email='mli@amazon.com',
    url='https://zh.d2l.ai',
    description='Dive into Deep Learning (in Chinese) Utils',
    long_description='Dive into Deep Learning Book (in Chinese) Utilities',
    license='Apache-2.0',

    # Package info
    packages=find_packages(exclude=('tests',)),

    zip_safe=True,
    install_requires=requirements,
)
