"""
TensorFlow-Bayes - Implementation of Bayesian models with TensorFlow Probability
See: https://github.com/huangyh09/TensorFlow-Bayes
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Set __version__ for the project.
exec(open("./TFBayes/version.py").read())

# Get the long description from the relevant file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()
    
reqs = ['numpy>=1.9.0', 'scipy>=1.0', 'matplotlib', 'scikit-learn',
        'tensorflow>=2.0.0', 'tensorflow-probability>=0.8.0']

setup(
    name='TensorFlow-Bayes',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=__version__,

    description='Bayesian models implementation with TensorFlow Probability',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/huangyh09/TensorFlow-Bayes',

    # Author details
    author='Yuanhua Huang',
    author_email='yuanhua@hku.hk',

    # Choose your license
    license='Apache-2.0',

    # What does your project relate to?
    keywords=['Bayesian model', 'variantional inference'],

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    
    install_requires=reqs,

    extras_require={
        'docs': [
            'sphinx_bootstrap_theme']},

    py_modules = ['TFBayes']

    # buid the distribution: python setup.py sdist
    # upload to pypi: twine upload dist/...
)