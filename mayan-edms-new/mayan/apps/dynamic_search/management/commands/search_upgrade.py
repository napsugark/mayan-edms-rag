from django.core.management.base import BaseCommand

from ...search_backends import SearchBackend


class Command(BaseCommand):
    help = 'Upgrade existing search backend persistent structures.'

    def handle(self, *args, **options):
        backend = SearchBackend.get_instance()

        backend.upgrade()
