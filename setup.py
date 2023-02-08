import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "open-retrieval",
    version = "0.1",
    author = "Corran McCallum & Christoph Schuhmann",
    author_email = "https://laion.ai/",
    description = "Retrieve semantically close text embeddings using a prebuilt FAISS index",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/orgs/LAION-AI/open-retrieval",
    project_urls = {
        "Bug Tracker": "https://github.com/orgs/LAION-AI/open-retrieval",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires = ">=3.6"
)