import re

# Change the content according to your package.
import setuptools

# Extract the version from the init file.
VERSIONFILE = "tsds/__init__.py"
getversion = re.search(
    r"^__version__ = ['\"]([^'\"]*)['\"]", open(VERSIONFILE, "rt").read(), re.M
)
if getversion:
    new_version = getversion.group(1)
else:
    raise RuntimeError(f"Unable to find version string in {VERSIONFILE}.")

# Configurations
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    install_requires=["matplotlib", "numpy", "pandas"],  # TODO deterine dependencies
    python_requires=">=3",  # Minimum Python version
    name="tsds",
    version=new_version,  # Version
    author="J.Udovic",  # Author name
    author_email="jakobudovic2@gmail.com",  # Author mail
    description="Python package for my library tsds.",  # Short package description
    long_description=long_description,  # Long package description
    long_description_content_type="text/markdown",
    url="https://github.com/jakobudovic/tsds",  # Url to your Git Repo
    download_url="https://github.com/jakobudovic/tsds"  # TODO release
    + new_version
    + ".tar.gz",
    packages=setuptools.find_packages(),  # Searches throughout all dirs for files to include
    include_package_data=True,  # Must be true to include files depicted in MANIFEST.in
    license_files=["LICENSE"],  # License file
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
)
