from django.apps import apps
from django.core.management.commands.makemigrations import Command as DjangoCommand

from ...literals import DJANGO_IGNORE_APP_MIGRATIONS


class Command(DjangoCommand):
    def handle(self, *app_labels, **options):
        if not app_labels:
            app_labels = {
                app_config.label for app_config in apps.get_app_configs()
            }
            app_labels -= DJANGO_IGNORE_APP_MIGRATIONS

        return super().handle(*app_labels, **options)
