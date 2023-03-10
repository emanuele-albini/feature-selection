__all__ = ['load_json', 'save_json']

import re
import os
import logging
import orjson
import datetime
from dateutil.parser import parse as datetime_parse
import numpy as np


def default_serializer(o):
    if isinstance(o, datetime.datetime):
        return dict(
            __type__='datetime.datetime',
            __value__=o.isoformat(),
        )
    elif isinstance(o, np.ndarray):
        return dict(
            __type__='np.ndarray',
            __dtype__=o.dtype.name,
            __value__=o.tolist(),
        )
    elif isinstance(o, np.integer):
        return int(o)
    elif isinstance(o, np.floating):
        return float(o)
    else:
        raise NotImplementedError('Unsupported type for JSON serialization: {}'.format(type(o)))


def deserialize_hook(o):
    if isinstance(o, list):
        return [deserialize_hook(x) for x in o]
    elif isinstance(o, dict):
        if '__type__' in o:
            if o['__type__'] == 'datetime.datetime':
                return datetime_parse(o['__value__'])
            elif o['__type__'] == 'np.ndarray':
                # We do not cast
                if '__dtype__' in o and re.match('^(float|int)[0-9]+$', o['__dtype__']) is not None:
                    return np.array(o['__value__'], dtype=getattr(np, o['__dtype__']))
                else:
                    return np.array(o['__value__'])
            else:
                raise NotImplementedError("Unsupported __type__")
        else:
            return {k: deserialize_hook(v) for k, v in o.items()}
    else:
        return o


def __save_json(obj, filename, serialize=False):

    with open(filename, "wb") as handle:
        if serialize:
            logging.info(f'Saving JSON with serialization to {filename}...')
            handle.write(
                orjson.dumps(
                    obj,
                    default=default_serializer,
                    option=orjson.OPT_PASSTHROUGH_DATETIME | orjson.OPT_PASSTHROUGH_DATACLASS
                )
            )
        else:
            logging.info(f'Saving JSON (without serialization) to {filename}...')
            handle.write(orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2))


def save_json(obj, filename, serialize=False, append=False):

    # Extension standardization
    if not filename.endswith('.json'):
        filename += '.json'

    if append and os.path.exists(filename):
        content = load_json(filename)
        if isinstance(content, dict) and isinstance(obj, dict):
            for k, v in obj.items():
                if k not in content or (content[k] == obj[k]):
                    content[k] = obj[k]
                else:
                    raise RuntimeError('Attempted override on JSON append.')

            # Save
            __save_json(content, filename, serialize=serialize)
        else:
            raise NotImplementedError(f'Append unsupported on this content type: {type(content)}.')

    else:
        return __save_json(obj, filename, serialize=serialize)


def load_json(filename):

    # Extension standardization
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, "rb") as handle:
        logging.info('Loading JSON from {}...'.format(filename))
        return deserialize_hook(orjson.loads(handle.read()))
