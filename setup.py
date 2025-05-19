from setuptools import setup, find_packages

setup(
    name='twinsight_model',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'numpy'
    ],
    entry_points={
        'console_scripts': [
            'twinsight-cli = cli:main'
        ]
    },
)
