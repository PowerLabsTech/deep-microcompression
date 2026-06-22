from setuptools import setup, find_packages

setup(
    name="deep_microcompression",
    version="0.1.0",
    packages=find_packages(
        exclude=["*.venv*", ".venv*", "deployment*", "*.pycache*", "__pycache__*"]
    ),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "numpy",
        "tqdm",
        "scipy",
        "google-cloud-storage",
    ],
)
