"""
    S3 General Utilities
"""

__author__ = "Emanuele Albini"
__all__ = [
    "S3Bucket",
]

import os
import sys
import glob
import threading
import datetime
import logging
import dateutil
from typing import Optional, Union, List
from itertools import chain
from pathlib import PurePath

import numpy as np
import pandas as pd
from tqdm import tqdm

import boto3
import botocore


class Progress(object):
    def __init__(self, key, size, name):
        self._lock = threading.Lock()
        self._tqdm = tqdm(desc=name + ' ' + key, total=float(size), unit='B', unit_scale=True)

    def __call__(self, bytes_amount):
        with self._lock:
            self._tqdm.n += bytes_amount
            self._tqdm.refresh()
            if self._tqdm.n >= self._tqdm.total:
                self._tqdm.close()


class ProgressUpload(Progress):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, name='Upload', **kwargs)


class ProgressDownload(Progress):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, name='Download', **kwargs)


class _Remote:
    def __init__(self, bucket: str):
        self.bucket = bucket

    def get_size(self, key):
        return boto3.client('s3').head_object(Bucket=self.bucket, Key=key)['ContentLength']

    def get_last_modified(self, key):
        return boto3.client('s3').head_object(Bucket=self.bucket, Key=key)['LastModified']

    def exists(self, key):
        try:
            boto3.client('s3').head_object(Bucket=self.bucket, Key=key)
        except botocore.exceptions.ClientError as e:
            # The key does not exists
            if e.response['Error']['Code'] == "404":
                return False
            else:
                # Something else has gone wrong.
                raise e
        return True

    def list(self,
             prefix: Optional[str] = None,
             recursive: bool = True,
             verbose=False) -> Union[List[dict], pd.DataFrame]:
        """List the contents of the (remote) S3 Bucket

        Args:
            prefix (str, optional): Path prefix
            frame (bool): If True, returns a DataFrame

        Returns:
            Union[List[dict]], pd.DataFrame] : S3 Bucket Contents as a list of dictionaries or a DataFrame
        """
        # Setup an S3 client
        s3 = boto3.client('s3')

        # Setup a paginator for 'list_objects'
        paginator = s3.get_paginator('list_objects')
        if prefix is None:
            prefix = ''
        else:
            prefix = PurePath(prefix).as_posix()
        page_iterator = paginator.paginate(Bucket=self.bucket, Prefix=prefix, Delimiter='' if recursive else prefix)

        # List the objects in the bucket
        try:
            contents_pages = [
                page['Contents'] for page in tqdm(page_iterator, 'Bucket Pages (List Objects)', disable=not verbose)
            ]
            contents = list(chain.from_iterable(contents_pages))
        except KeyError as e:
            # Nothing to list (empty)
            if e.args[0] == 'Contents':
                return []
            else:
                raise e

        contents = [
            {key: value
             for key, value in content.items() if key in ['Key', 'LastModified', 'Size']}
            for content in contents if not content['Key'].endswith('/')
        ]

        return contents

    def get_bucket(self):
        return boto3.resource('s3').Bucket(self.bucket)


class _Local:
    def __init__(self, local_directory: str):
        """
        Args:
            local_directory (str) : Directory where to save files
        """
        self.local_directory = local_directory

    def key_to_local_uri(self, key: str):
        return f"{self.local_directory}/{key}"

    @staticmethod
    def _listdir(path, key=None, recursive=True):
        if key is None:
            key = ''

        # No file to list of the path does not exists
        if not os.path.exists(path):
            return []

        contents = os.listdir(path)
        contents_paths = [os.path.join(path, c) for c in contents]
        contents_keys = [PurePath(os.path.join(key, c)).as_posix() for c in contents]

        if not recursive:
            return contents_keys
        else:
            return contents_keys + list(
                chain.from_iterable(
                    [_Local._listdir(path=p, key=k) for k, p in zip(contents_keys, contents_paths) if os.path.isdir(p)]
                )
            )

    @staticmethod
    def _filter(key):
        s = key.split('/')
        for s_ in s:
            # Hide Python cache
            if s_.startswith('__'):
                return True
            # Hide hidden
            if s_.startswith('.'):
                return True
        return False

    def _filter_keys(self, keys):
        return [key for key in keys if not _Local._filter(key) and not os.path.isdir(self.key_to_local_uri(key))]

    def list(self, prefix: Optional[str] = None, recursive: bool = True) -> Union[List[str], pd.DataFrame]:
        """List the contents of the local cache of the S3 bucket

        Args:
            prefix (str, optional): Path prefix
            frame (bool): If True, returns a DataFrame

        Returns:
            Union[List[dict]], pd.DataFrame] : Contents of the cache as a list of dictionaries or a DataFrame
        """

        if prefix is not None:
            prefix = PurePath(prefix).as_posix()

        # List recursively the directory
        path = os.path.join(self.local_directory, prefix) if prefix is not None else self.local_directory
        keys = [key for key in _Local._listdir(path, key=prefix, recursive=recursive)]

        # Filter invalid keys
        keys = self._filter_keys(keys)

        contents = [
            {
                'Key': key,
                'LastModified': self.get_last_modified(key),
                'Size': self.get_size(key),
            } for key in keys
        ]

        return contents

    def get_last_modified(self, key):
        return datetime.datetime.fromtimestamp(os.path.getmtime(self.key_to_local_uri(key)), dateutil.tz.tz.tzutc())

    def get_size(self, key):
        return os.path.getsize(self.key_to_local_uri(key))

    def exists(self, key):
        return os.path.exists(self.key_to_local_uri(key))

    def set_last_modified(self, key, date):
        epoch = date.timestamp()
        os.utime(self.key_to_local_uri(key), (epoch, epoch))


class S3Bucket:
    """This class can be used to manage and S3 Bucket. It allows for:
        - Listing the contents
        - Filtering the contents only in a specific directory
        - Accessing files in the bucket
        - Managing a local cache of the bucket
    """
    def __init__(self, bucket: str, local_directory: str = './bucket'):
        """
        Args:
            bucket (str): Bucket ID
            local_directory (str) : Directory where to cache downloaded files
        """
        self.local = _Local(local_directory=local_directory)
        self.remote = _Remote(bucket=bucket)

    @staticmethod
    def __list_to_frame(contents):
        # Create a DataFrame
        if len(contents) > 0:
            df = pd.DataFrame(contents)
        else:
            df = pd.DataFrame([], columns=['Key', 'LastModified', 'Size'])

        # Sort by size
        df = df.sort_values(['LastModified', 'Key'], ascending=False)

        # Remove directories
        # df = df[df['Key'].apply(lambda path: not path.endswith('/'))]

        return df.set_index('Key')

    @staticmethod
    def _remove_prefix(content, prefix):
        if prefix is not None:
            if not prefix.endswith('/'):
                prefix = prefix + '/'
            if content['Key'].startswith(prefix):
                content['Key'] = content['Key'][len(prefix):]
        return content

    @staticmethod
    def _merge_contents(contents, additional_contents):

        # Tranform to dictionary for manipulation
        contents = {content['Key']: content for content in contents}

        # Set the location filed to local
        for key in contents:
            contents[key]['Location'] = 'Local'
            contents[key]['Sync'] = False
            contents[key]['LastModifiedLocal'] = contents[key]['LastModified']
            contents[key]['LastModifiedRemote'] = '-'

        # Add remote contents that are not already in there
        for content in additional_contents:
            key = content['Key']
            if key not in contents or (key in contents and contents[key]['LastModified'] < content['LastModified']):
                content['Location'] = 'Remote'
                content['Sync'] = False
                content['LastModifiedRemote'] = content['LastModified']
                content['LastModifiedLocal'] = contents.get(key, {'LastModified': '-'})['LastModified']
                contents[key] = content

            if key in contents and contents[key]['LastModified'] == content['LastModified']:
                contents[key]['Sync'] = True

        # Transform back to a list
        contents = list(contents.values())

        return contents

    def list(
        self,
        prefix: Optional[str] = None,
        frame: bool = False,
        local_only: bool = False,
        remote_only: bool = False,
        remove_prefix: bool = False,
        recursive: bool = True,
    ) -> Union[List[dict], pd.DataFrame]:
        """List the contents of the bucket
            If the remote bucket is available it will list the remote bucket.
            Otherwise it will fall back to the local cache (raising a warning).

        Returns:
            Union[List[dict], pd.DataFrame]: List of files available.
        """

        if local_only:
            contents = self.local.list(prefix=prefix, recursive=recursive)
        elif remote_only:
            contents = self.remote.list(prefix=prefix, recursive=recursive)
        else:
            contents = S3Bucket._merge_contents(
                self.local.list(prefix=prefix, recursive=recursive),
                self.remote.list(prefix=prefix, recursive=recursive)
            )

        if remove_prefix:
            contents = [S3Bucket._remove_prefix(content) for content in contents]

        if frame:
            contents = self.__list_to_frame(contents)

        return contents

    def local_time_delta(self, key):
        if not self.local.exists(key) and not self.remote.exists(key):
            return datetime.timedelta(0)
        elif self.local.exists(key) and not self.remote.exists(key):
            return datetime.timedelta(weeks=52 * 100)
        elif self.remote.exists(key) and not self.local.exists(key):
            return datetime.timedelta(weeks=-52 * 100)
        else:
            return self.local.get_last_modified(key) - self.remote.get_last_modified(key)

    def is_local_more_recent(self, key):
        return self.local_time_delta(key) > datetime.timedelta()

    def is_remote_more_recent(self, key):
        return self.local_time_delta(key) < datetime.timedelta()

    def _get(self, key):
        local_uri = self.local.key_to_local_uri(key)

        # Download
        self.remote.get_bucket().download_file(
            key, local_uri, Callback=ProgressDownload(key, self.remote.get_size(key))
        )

        # Set the correct last modified
        self.local.set_last_modified(key, self.remote.get_last_modified(key))

    def get(self, key: str, local_only=False, uri_only=False, force=False) -> str:
        """Return the URI of a file to be consumed
            If the file is not available locally, it downloads it

        Args:
            key (str): The key of the file
            local_only (bool, optional): If True, it does not download the file.

        Returns:
            str: Local File URI
        """

        assert not local_only or not force, "Cannot force download if local_only = True."
        assert not uri_only or not force, "Cannot force download if uri_only = True."

        uri = self.local.key_to_local_uri(key)

        if uri_only:
            return uri

        if not local_only and (force or self.is_remote_more_recent(key)):
            os.makedirs(os.path.dirname(uri), exist_ok=True)
            self._get(key)

        return uri

    def _put(self, key):
        local_uri = self.local.key_to_local_uri(key)

        # Upload the file
        self.remote.get_bucket().upload_file(local_uri, key, Callback=ProgressUpload(key, self.local.get_size(key)))

        # We update the local modified date
        # Otherwise the file will be downloaded again at the following access
        self.local.set_last_modified(key, self.remote.get_last_modified(key))

    def put(self, key, force=False):
        if force or self.is_local_more_recent(key):
            self._put(key)

    def push(self, prefix=None, **kwargs):
        for key in self.list(prefix, frame=True).index.values:
            self.put(key, **kwargs)

    def pull(self, prefix=None, **kwargs):
        for key in self.list(prefix, frame=True).index.values:
            self.get(key, **kwargs)
