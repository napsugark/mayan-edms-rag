from pathlib import Path

from django.conf import settings

DEFAULT_SOURCES_BACKEND_ARGUMENTS = {}

DOCUMENT_FILE_SOURCE_METADATA_BATCH_SIZE = 100

DEFAULT_SOURCES_CACHE_MAXIMUM_SIZE = 100 * 2 ** 20  # 100 Megabytes
DEFAULT_SOURCES_CACHE_STORAGE_BACKEND = 'django.core.files.storage.FileSystemStorage'
DEFAULT_SOURCES_CACHE_STORAGE_BACKEND_ARGUMENTS = {
    'location': str(
        Path(settings.MEDIA_ROOT, 'source_cache')
    )
}

DEFAULT_SOURCES_LOCK_EXPIRE = 600

ERROR_LOG_DOMAIN_NAME = 'sources'

SOURCE_ACTION_EXECUTE_TASK_PATH = 'mayan.apps.sources.tasks.task_source_backend_action_execute'

STORAGE_NAME_SOURCE_CACHE_FOLDER = 'sources__source_cache'
