from setuptools import setup, find_packages
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path('ssvm/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name="ssvm",  # TODO: This name is already in PyPi. We need to choose something different.
    version=main_ns["__version__"],
    license="MIT",
    packages=find_packages(exclude=["results*", "tests", "examples", "*.ipynb"]),

    # Minimum requirements the package was tested with
    install_requires=[
        "numpy",
        "scikit-learn",
        "pandas",
        "scipy",
        "joblib",
        "pip",
        "matchms",
        "more-itertools",
        "tqdm",
        "tbb",
        "numba",
        "networkx"
    ],

    # Metadata
    author="Eric Bach",
    author_email="eric.bach@aalto.fi",
    description="Structure Support Vector Machine (SSVM) implementation for (MS, RT)-sequence identification.",
    url="https://github.com/aalto-ics-kepaco/msms_rt_ssvm",
)
