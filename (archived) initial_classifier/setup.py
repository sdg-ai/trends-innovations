# -*- coding: utf-8 -*-

import setuptools
from setuptools.command.install import install as _install


# Override default install
class CustomInstall(_install):
    def run(self):
        # Install required modules
        _install.do_egg_install(self)

        # Download nltk models
        import nltk
        nltk.download("punkt")
        nltk.download("vader_lexicon")
        nltk.download("wordnet")

        # Download spacy language model
        import spacy.cli
        spacy.cli.download("en_core_web_lg")


with open("version.txt", "r") as fh:
    version = fh.read().strip()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    cmdclass={'install': CustomInstall},
    name="trends_innovation",
    version=version,
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    packages=setuptools.find_packages(),
    scripts=[],

    # External libraries
    install_requires=["numpy==1.19.2", "pandas", "scikit-learn", "pytest", "pyyaml", "spacy"],

    setup_requires=['spacy', 'nltk'],

    package_data={
        # If any package contains these extensions, include them:
        '': ['*.txt', '*.ods', '*.xlsx', '*.tsv', '*.csv', '*.pkl',
             'data/*', 'data/*/*', 'data/*/*/*', 'data/*/*/*/*', 'data/*/*/*/*/*',
             'data/*.*', 'data/*/*.*', 'data/*/*/*.*', 'data/*/*/*/*.*',  'data/*/*/*/*/*.*',
             ],
    },
)
