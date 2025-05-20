from setuptools import setup, find_packages

setup(
    name='twinsight_model',
    version='0.1.0',
    author='Lakshmi Anandan',
    author_email='lakshmi19anandan@gmail.com',
    description='A predictive modeling package for personalized health risk estimation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Lakshmi2819/Regression-Model',
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
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
