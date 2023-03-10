from typing import Union, List


def _add_extension_to_filename(filename, format):
    if format.startswith('.'):
        format = format[1:]
    if not filename.endswith(f'.{format}'):
        filename += f'.{format}'
    return filename


def _remove_extension_from_filename(filename, format: Union[str, List[str]] = None, formats=None):
    # Alias
    if format is None:
        format = formats
    if format is None:
        raise ValueError('format (or formats) must be passed.')

    if isinstance(format, str):
        format = [format]
    for format_ in format:
        if filename.endswith(f'.{format}'):
            filename = filename[:-(len(format) + 1)]
            break
    return filename


def _infer_format(filename, formats, default=None):
    for format_ in formats:
        if filename.endswith(f'.{format_}'):
            return format_
    return default
