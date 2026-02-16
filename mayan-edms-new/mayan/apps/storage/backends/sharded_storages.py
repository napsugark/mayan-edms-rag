from pathlib import Path

from django.core.files.storage import FileSystemStorage
from django.utils._os import safe_join


class ShardedDirectoryFileSystemStorageMixin:
    DEFAULT_SHARDING_LEVELS = 2

    def __init__(self, *args, sharding_levels=None, **kwargs):
        if sharding_levels is None:
            sharding_levels = self.DEFAULT_SHARDING_LEVELS

        self.sharding_levels = sharding_levels

        super().__init__(*args, **kwargs)

    def get_sharded_name(self, name):
        path_original = Path(name)

        filename = path_original.name

        parts = [
            filename[:level] for level in range(1, self.sharding_levels + 1)
        ]

        path_new = Path(*parts, filename)

        return path_new

    def get_sharded_path(self, name):
        sharded_name = self.get_sharded_name(name=name)

        result = safe_join(self.location, sharded_name)

        return result


class ShardedDirectoryFileSystemStorage(
    ShardedDirectoryFileSystemStorageMixin, FileSystemStorage
):
    def _save(self, name, content):
        get_sharded_name = self.get_sharded_name(name=name)
        return super()._save(content=content, name=get_sharded_name)
