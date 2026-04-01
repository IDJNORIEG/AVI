from setuptools import setup, find_packages

setup(
    name='your_package_name',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A brief description of your package',
    packages=find_packages(),
    install_requires=[
        # List your package dependencies here
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)