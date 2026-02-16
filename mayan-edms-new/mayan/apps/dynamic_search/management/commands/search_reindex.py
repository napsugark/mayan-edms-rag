from django.core.management.base import BaseCommand

from ...tasks import task_reindex_backend


class Command(BaseCommand):
    help = 'Erases and populates the search backend internal indexes.'

    def handle(self, *args, **options):
        task_reindex_backend.apply_async()
