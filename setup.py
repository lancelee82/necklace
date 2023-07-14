import setuptools


packages = [
    'necklace',
    'necklace.rpc',
    'necklace.cuda',
    'necklace.data',
    'necklace.frmwrk',
    'necklace.trainer',
    'necklace.utils',
]

setuptools.setup(
    name='necklace',
    version='0.5.2',
    author="Lance Lee",
    author_email="lancelee82@163.com",
    description="necklace - distributed deep learning framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT License",
    url="https://github.com/lancelee82/necklace",
    packages=packages,
    install_requires=[
        'numpy', 'numba', 'pynccl',
        'zerorpc', 'msgpack==0.5.6', 'msgpack-numpy'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
