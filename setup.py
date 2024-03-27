from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name='micro_agent',
    version='0.1',
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=required
)
