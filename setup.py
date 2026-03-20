from setuptools import setup, find_packages

setup(
    name="echoloc-nn",
    version="0.1.0",
    description="Neural echolocation — ChirpSignal, EchoSimulator, ChirpEncoder, TransformerLocator",
    author="Daniel Schmidt",
    author_email="danschmidt88@gmail.com",
    url="https://github.com/danieleschmidt/echoloc-nn",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
