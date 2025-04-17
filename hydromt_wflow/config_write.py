"""Function to handle config parameters that are written as 'key.value'."""

from functools import reduce


def _update_nested_dict(key: str, value, data: dict):
    keys = key.split(".")
    values_dict = reduce(lambda d, k: d.setdefault(k, {}), keys[:-1], data)[
        keys[-1]
    ].copy()
    for k, v in values_dict.items():
        if ".value" in k:
            value.update({k: v})
    reduce(lambda d, k: d.setdefault(k, {}), keys[:-1], data)[keys[-1]] = value


def _get_key_values_to_move(d, parent_key="", keys_to_move=[]):
    if isinstance(d, dict):
        for key, value in d.items():
            if key == "value":
                keys_to_move.append((parent_key, value))
            elif isinstance(value, dict):
                _get_key_values_to_move(
                    value, parent_key + "." + key if parent_key else key, keys_to_move
                )

    if keys_to_move is not None:
        return keys_to_move


def _update_config_dict_with_value_keys(keys_values_to_move: tuple, data: dict):
    for parent, value in keys_values_to_move:
        parents, parent = parent.rsplit(".", maxsplit=1)
        value_to_set = {parent + ".value": value}
        _update_nested_dict(key=parents, value=value_to_set, data=data)


def handle_key_dot_value_config_items(data_dict: dict) -> dict:
    """Handle config parameters that are written as 'key.value'.

    This function checks if there are any keys in the dictionary that are named value
    and sets those key values a level higher in the nested dictionary. For instance,
    the following dictionary:
    {"input":
        "vertical": {
            "ksathorfrac": {
                "value": 100
                }
            }
        }
    will become:
    {"input":
        "vertical": {
            "ksathorfrac.value": 100
            }
        }


    """
    keys_values_to_move = _get_key_values_to_move(data_dict)
    _update_config_dict_with_value_keys(
        keys_values_to_move=keys_values_to_move, data=data_dict
    )
    return data_dict
