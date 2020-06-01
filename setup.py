from setuptools import setup, find_packages

setup(
    name="ssvm",
    version="0.1.0",
    license="MIT",
    packages=find_packages(exclude=["results*", "tests", "examples", "*.ipynb"]),

    # Minimum requirements the package was tested with
    install_requires=[
        "numpy",
        "scikit-learn",
        "pandas"
    ],

    # Metadata
    author="Eric Bach",
    author_email="eric.bach@aalto.fi",
    description="Structure SVM implementation for (MS, RT)-sequence identification.",
    url="https://github.com/aalto-ics-kepaco/msms_rt_ssvm",
)