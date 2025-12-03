#*********************************************************************
# Author           : Libin
# Company          : huz.zj.yc
# Last modified    : 2025-12-03 10:40
# Filename         : utilities.py
# Project          : HuRAG/embedding-service
#*********************************************************************

def dict_to_namespace(data):
    """
    Recursively converts a dictionary (and nested dicts/lists of dicts)
    into a types.SimpleNamespace object.
    """
    from types import SimpleNamespace

    if isinstance(data, dict):
        # Convert dictionary items recursively
        return SimpleNamespace(
            **{key: dict_to_namespace(value) for key, value in data.items()}
        )

    elif isinstance(data, list):
        # Convert list items if they are dictionaries
        return [dict_to_namespace(item) for item in data]

    else:
        # Return all other types (strings, integers, etc.) unchanged
        return data

