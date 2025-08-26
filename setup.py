from setuptools import setup, find_packages

setup(
    name='twinsight_model',
    version='0.2.0',
    author='Lakshmi Anandan, Shawn ONeil',
    author_email='lakshmi19anandan@gmail.com, shawn@tislab.org',
    description='A predictive modeling package for personalized health risk estimation leveraging All of Us data.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Lakshmi2819/Regression-Model',
    packages=find_packages(),
    install_requires=[
        # Only non-AoU dependencies - AoU provides pandas, numpy, scikit-learn, pyyaml, joblib
        'lifelines==0.29.0',  # Pinned for pandas 2.0.3 compatibility
        'google-cloud-bigquery>=2.34.4',
        'protobuf>=3.20.0,<4.0.0',
    ],
    entry_points={
        'console_scripts': [
            'twinsight-cli = cli:main'
        ]
    }, 
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
