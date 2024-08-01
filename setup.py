import setuptools

with open("README.md", 'r') as readme:
    long_description = readme.read()

with open("requirements.txt", 'r') as dependencies:
    requirements = [pkg.strip() for pkg in dependencies]

setuptools.setup(
    name="matryoshka",
    version="0.2.17.dev0",
    author="Jamie Donald-McCann",
    author_email="jamie.donald-mccann@port.ac.uk",
    description="Emulated galaxy power spectrum",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JDonaldM/Matryoshka",
    packages=setuptools.find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    install_requires=requirements,
    python_requires=">=3.8",
)
