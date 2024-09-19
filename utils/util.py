import json
import os.path


def write_json(json_obj, json_path, update=False):

    if update:
        if os.path.exists(json_path):
            existing_data = load_json(json_path)
            if isinstance(existing_data, list):
                existing_data.extend(json_obj)
            elif isinstance(existing_data, dict):
                existing_data.update(json_obj)
            else:
                raise ValueError("Wrong formats")

            json_obj = existing_data

    with open(json_path, "w+") as f:
        json.dump(json_obj, f)


def load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)
