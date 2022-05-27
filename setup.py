from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="image_stack", # Replace with your own username
    version="0.0.1",
    author="Phillip Manley",
    author_email="manley@zib.de",
    description="processing of stacks of images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://github.com/nano-sippe/dispersion",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy', 'matplotlib', 'scipy'],
    python_requires='>=3.6',    
)

#data_files=[('config',['cfg/config.yaml'])],
