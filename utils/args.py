from argparse import Namespace

import yaml


def parse_args(parser, **kwargs):
    """Parse arguments and return them as a Namespace.
        It has the following priorities (highest to lowest):
        1. **kwargs
        2. kwargs['config'] = FILENAME.yaml
        3. argparse --config FILENAME.yaml
        4. argparse --key value
    
        Any argument can be passed in any of the three ways.
        The higher priority source will overwrite the lower priority source.

    Args:
        parser (parse.ArgumentParser): Argument parser.

    Returns:
        argparse.Namespace: Arguments.
    """

    # Add config parameter
    if parser is not None:
        if 'config' not in [action.dest for action in parser._actions]:
            parser.add_argument('--config', type=str)

        # Get the dictionary (argparse --key value)
        args, unknown = parser.parse_known_args()
        args = vars(args)
    else:
        args = dict()

    # Jinja clean-up if not used
    if kwargs.get('config', None) == "{{CONFIG}}":
        del kwargs['config']

    # Config in kwargs has higher priority of than argparse
    if 'config' in kwargs:
        args['config'] = kwargs['config']

    # YAML Config file
    if 'config' in args and args['config'] is not None:
        with open(args['config'], 'r') as f:
            config_args = yaml.safe_load(f)
            assert 'config' not in config_args, 'Config file cannot contain "config" key.'
            args.update(config_args)

    # kwargs has the highest priority
    args.update(kwargs)

    # Handling of None strings > values
    for key, value in args.items():
        if value == 'None':
            print(value, key)
            args[key] = None

    # Printing
    if not 'verbose' in args or args['verbose'] > 0:
        print(args)

    return Namespace(**args)
