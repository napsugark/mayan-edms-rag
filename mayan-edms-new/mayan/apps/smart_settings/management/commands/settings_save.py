from django.core import management

from ...settings import setting_cluster


class Command(management.BaseCommand):
    help = 'Save the current settings into the configuration file.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--filepath', action='store', dest='filepath',
            help='Filename and path where to save the configuration file.'
        )

    def handle(self, *args, **options):
        filepath = options.get('filepath')
        setting_cluster.do_configuration_file_save(path=filepath)
