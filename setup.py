from setuptools import setup, find_packages
import os

__version__ = '1.0.0'
url = 'https://github.com/ta-akb/proms'

install_requires = [
    'scikit-survival>=0.24.1',
    'numpy>=2.2.4',
    'scipy>=1.15.2',
    'scikit-learn>=1.6.1',
    'pandas>=2.2.3',
    'xgboost>=3.0.0',
    'matplotlib>=3.10.0',
    'pyyaml>=6.0.2',
    'seaborn>=0.13.2'
]


setup(
    name='proms3',
    version=__version__,
    description='Protein Markers Selection 3 algorithmsy',
    author='Taiki Akiba',
    author_email='@gmail.com',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=[
        'machine learning', 'feature selection', 'proteomics',
        'multiomics', 'biomarker', 'genomics', 'bioinformatics'
    ],
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.11',
    install_requires=install_requires,
    packages=find_packages(include=['proms3.*']),
    #packages=find_packages(),
    entry_points={
        "console_scripts": [
        "proms3=proms.__main__:cli_main",
        "proms3_train=proms.__main__:main",
        "proms3_predict=proms.predict:main"
        ]
    }
    #entry_points={"console_scripts": ["proms_train=proms.__main__:main", 
    #              "proms_predict=proms.predict:main"]}
)
