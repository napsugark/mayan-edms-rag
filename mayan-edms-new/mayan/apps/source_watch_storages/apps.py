from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig


class SourceWatchStorageApp(MayanAppConfig):
    app_namespace = 'source_watch_storages'
    app_url = 'source_watch_storages'
    has_tests = True
    name = 'mayan.apps.source_watch_storages'
    verbose_name = _(message='Watch storages')
