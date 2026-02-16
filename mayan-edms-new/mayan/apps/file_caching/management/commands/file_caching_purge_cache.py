from django.core import management

from ...models import Cache


class Command(management.BaseCommand):
    help = 'Purge the selected cache directly.'
    missing_args_message = 'You must provide a defined storage name that corresponds to a cache.'

    def add_arguments(self, parser):
        parser.add_argument(
            dest='storage_name', help='Name of the cache defined storage.',
            metavar='<storage name>'
        )

    def handle(self, storage_name, **options):
        try:
            cache = Cache.objects.get(defined_storage_name=storage_name)
        except Cache.DoesNotExist:
            self.stderr.write(
                msg='Unknown cache storage `{}`'.format(storage_name)
            )
            exit(1)

        self.stdout.write(
            msg='\nPurging cache "{}"...'.format(cache)
        )

        cache.purge(user=None)

        self.stdout.write(
            msg='\nCache "{}", purged.'.format(cache)
        )
