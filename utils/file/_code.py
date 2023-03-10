import os
from argparse import ArgumentParser
import shutil


def save_code(directories, path, skip_main=False):

    if path.endswith('/'):
        path = path[:-1]

    if os.path.exists(path):
        shutil.rmtree(path)

    if os.path.exists(path + '.tar.gz'):
        os.remove(path + '.tar.gz')

    os.makedirs(path, exist_ok=True)

    assert os.path.isdir(path), "Code must be saved in a directory, not a file."

    # Save the main script
    if not skip_main:
        parser = ArgumentParser()
        shutil.copyfile(parser.prog, os.path.join(path, parser.prog))

    # Copy the code
    if directories is not None:
        for directory in directories:
            if os.path.isdir(directory):
                shutil.copytree(directory, os.path.join(path, os.path.basename(directory)))
            else:
                shutil.copyfile(directory, os.path.join(path, os.path.basename(directory)))

    # Put all the code in an archive
    shutil.make_archive(path, 'gztar', path)
    shutil.rmtree(path)
