from argparse import Namespace
import yaml


def save_yaml(obj, path):
    if not path.endswith('.yaml') and not path.endswith('.yml'):
        path += '.yaml'
    with open(path, 'w') as f:
        # Trasform NameSpace to dictionary
        if isinstance(obj, Namespace):
            obj = vars(obj)
        yaml.safe_dump(obj, f)


def load_yaml(obj, path):
    if not path.endswith('.yaml') and not path.endswith('.yml'):
        path += '.yaml'
    with open(path, 'r') as f:
        return yaml.safe_load(f)
