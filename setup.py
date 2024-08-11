# setup.py

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

setup(
    name="my_project",
    version="0.1",
    description="A description of your project",
    author="Your Name",
    author_email="hyli@brandeis.edu",
    url="https:github.com/hyli999/Livestock-Risk-Protection-Hedging-Strategy",
    packages=find_packages(),
    install_requires=required_packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Specify Python version if needed
)