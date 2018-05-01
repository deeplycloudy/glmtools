from setuptools import setup, find_packages

setup(
    name='glmtools',
    version='0.1dev',
    description='Python tools for reading, processing, and visualizing GOES Geostationary Lightning Mapper data',
    packages=find_packages(),# ['glmtools',],
    author='Eric Bruning',
    author_email='eric.bruning@gmail.com',
    url='https://github.com/deeplycloudy/glmtools/',
    license='BSD-3-Clause',
    long_description=open('README.md').read(),
    include_package_data=True,
)
