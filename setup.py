from setuptools import setup, find_packages

with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()

with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name='HiTIN',
    version='0.1.20241002-5',
    description='A packaged version of the HiTIN model implemented by Zhu et al.',
    long_description=readme,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    url='https://github.com/beansrowning/HiTIN',
    author='Sean Browning',
    author_email='sbrowning@cdc.gov',
    license='MIT',
    packages=find_packages(include=["hi_tin", "hi_tin.*"]),
    install_requires=requirements,
    zip_safe=False
)