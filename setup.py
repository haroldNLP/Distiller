from setuptools import setup, find_packages

setup(
    name = "distiller",
    version = "1.0",
    keywords = ("distiller", "knowledge distillation"),
    description = "eds sdk",
    long_description = "eds sdk for python",
    license = "MIT Licence",

    url = "http://test.com",
    author = "test",
    author_email = "test@gmail.com",

    packages=["Distiller"],
    package_dir={'': 'src'},
    include_package_data = True,
    platforms = "any",
    install_requires = [
        "scipy >= 1.0.0",
        "numpy>=1.17",
        "requests>=2.22.0",
        "Cython",
        "fsspec",
        "wheel",
        "dill",
        "pandas",
        "tqdm>=4.27",
        "xxhash",
        "multiprocess",
        "packaging",
        "regex!=2019.12.17",
        "tensorboard",
        "sacremoses",
        "pyemd",
        "sklearn",
        "fairseq",
        'sentencepiece',
        "fastBPE",
        "boto3",
        "nlpaug==1.1.3",
        "transformers==4.3.2",
        "datasets"
    ],

    scripts = [],
    entry_points = {
        'console_scripts': [
            'test = test.help:main'
        ]
    }
)