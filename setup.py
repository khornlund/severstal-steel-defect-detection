import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())

requirements = [
    # use environment.yml
]

setup(
    name="sever",
    version="0.0.1",
    url="https://github.com/khornlund/severstal-steel-defect-detection",
    license='MIT',
    author="Karl Hornlund",
    author_email="karlhornlund@gmail.com",
    description="https://www.kaggle.com/c/severstal-steel-defect-detection/overview",
    long_description=read("README.rst"),
    packages=find_packages(exclude=('tests',)),
    entry_points={
        'console_scripts': [
            'sever=sever.cli:cli'
        ]
    },
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6'
    ],
)
