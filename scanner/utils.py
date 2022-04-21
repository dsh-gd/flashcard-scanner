# scanner/utils.py
# Utility functions.

import json
from typing import Dict


def load_dict(filepath: str) -> Dict:
    """Load a dictionary from a JSON's filepath.

    Args:
        filepath (str): JSON's filepath.

    Returns:
        A dictionary with the data loaded.
    """
    with open(filepath) as fp:
        d = json.load(fp)
    return d


def save_dict(
    d: Dict, filepath: str, ensure_ascii: bool = False, sortkeys: bool = False
) -> None:
    """Save a dictionary to a specific location.

    Args:
        d (Dict): A dictionary to save.
        filepath (str): A location to save the dictionary to as a JSON file.
        ensure_ascii (bool, optional): Ensure the output is valid ascii characters. Defaults to False.
        sortkeys (bool, optional): Sort keys in dict alphabetically. Defaults to False.
    """
    with open(filepath, "w") as fp:
        json.dump(
            d, fp=fp, ensure_ascii=ensure_ascii, indent=2, sort_keys=sortkeys
        )
