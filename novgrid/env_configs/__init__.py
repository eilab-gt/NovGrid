from typing import List, Dict, Any

import os.path
import json


def get_env_configs(name: str) -> List[Dict[str, Any]]:
    fname = f"{name}.json" if ".json" not in name else name
    full_fname = os.path.join(os.path.dirname(__file__), "json", fname)
    with open(full_fname) as f:
        return json.load(f)
