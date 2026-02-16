from django.apps import apps
from django.db.utils import OperationalError, ProgrammingError

from .literals import STORAGE_NAME_SOURCE_CACHE_FOLDER


def callback_update_source_cache_size(setting):
    Cache = apps.get_model(app_label='file_caching', model_name='Cache')

    try:
        cache = Cache.objects.get(
            defined_storage_name=STORAGE_NAME_SOURCE_CACHE_FOLDER
        )
    except (Cache.DoesNotExist, OperationalError, ProgrammingError):
        """
        Non fatal. Non initialized installation. Ignore exception.
        This will generate an error log entry similar to this.
        2023-12-12 09:12:46.925 UTC [79] ERROR:  relation "file_caching_cache" does not exist at character 121
        2023-12-12 09:12:46.925 UTC [79] STATEMENT:  SELECT "file_caching_cache"."id", "file_caching_cache"."defined_storage_name", "file_caching_cache"."maximum_size" FROM "file_caching_cache" WHERE "file_caching_cache"."defined_storage_name" = 'sources__source_cache' LIMIT 21
        """
    else:
        cache.maximum_size = setting.value
        cache.save()
