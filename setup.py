from pathlib import Path

from setuptools import setup

BASE_DIR = Path(__file__).parent

with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [line.strip() for line in file.readlines()]

test_packages = []

dev_packages = [
    "black==21.6b0",
    "flake8==3.9.2",
    "isort==5.9.1",
]

setup(
    name="flashcard-scanner",
    version="0.1",
    license="MIT",
    description="",
    author="Dmytro Shurko",
    author_email="",
    url="",
    python_requires=">=3.6",
    install_requires=[required_packages],
    extras_require={
        "test": test_packages,
        "dev": test_packages + dev_packages,
    },
    entry_points={
        "console_scripts": [
            "scanner = app.cli:app",
        ],
    },
)
