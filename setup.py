from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

setup(
    name='DAEpy',
    version='0.1',
    description='Collection of solvers for differential algebraic equations',
    author='William Price',
    author_email='wcp23@cam.ac.uk',
    packages=find_packages(),
    install_requires=[
    'numpy>=1.11.1',
    'scipy>=0.18.1']
)
