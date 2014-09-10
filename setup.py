# Euclidian's setup.py

from distutils.core import setup

version = 0.8

with open('README.txt', 'r') as readme:
    long_description = readme.read()

setup(
    name = "euclidian",
    packages = ["euclidian"],
    version = "{version}".format(version=version),
    description = "Simple 2D and 3D geometric primitives and operations",
    author = "Robert Smallshire",
    author_email = "rob@sixty-north.com",
    url = "http://code.sixty-north.com/euclidian",
    #download_url="".format(version=version),
    keywords = ["Python", "geometry"],
    license="MIT License",
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.4",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
        ],
    requires = [],
    long_description = long_description
)
