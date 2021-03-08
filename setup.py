from setuptools import setup, find_packages

setup(
    name='CAT',
    version='0.0.1',
    author='nnnyt',
    author_email='ningyt@mail.ustc.edu.cn',
    packages=find_packages(),
    url='https://github.com/bigdata-ustc/CAT',
    install_requires=[
        'torch',
        'vegas',
        'numpy',
        'scikit-learn',
    ],  # And any other dependencies foo needs
    entry_points={
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ]
)