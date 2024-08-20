from setuptools import setup, find_packages

setup(
    name='HiTIN',
    version='0.1',
    description='A packaged version of the HiTIN model implemented by Zhu et al.',
    python_requires=">=3.9",
    url='https://github.com/beansrowning/HiTIN',
    author='Sean Browning',
    author_email='sbrowning@cdc.gov',
    license='MIT',
    packages=find_packages(include=["hi_tin", "hi_tin.*"]),
    zip_safe=False
)