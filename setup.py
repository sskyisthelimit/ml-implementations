from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

setup(
    name='ml_implementations',
    version='0.1',
    packages=find_packages(),
    install_requires=required_packages,
    include_package_data=True,
    description='A showcase of ML implementations',
    author='Yalanskyi Roman',
    url='https://github.com/sskyisthelimit/ml-implementations',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
