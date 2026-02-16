from django.apps import apps
from django.db.utils import OperationalError, ProgrammingError

from .literals import STORAGE_NAME_WORKFLOW_CACHE


def callback_update_workflow_image_cache_size(setting):
    Cache = apps.get_model(app_label='file_caching', model_name='Cache')

    try:
        cache = Cache.objects.get(
            defined_storage_name=STORAGE_NAME_WORKFLOW_CACHE
        )
    except (OperationalError, ProgrammingError):
        """
        Non fatal. Non initialized installation. Ignore exception.
        """
    else:
        cache.maximum_size = setting.value
        cache.save()
