#!/usr/bin/env python

import configparser
import os
import optparse
from pathlib import Path
import sys

import sh

import django
from django.apps import apps
from django.conf import settings

TRANSIFEX_LANGUAGE_MAPPING = {
    'zh-Hans': 'zh_Hans'
}
PROJECT_LANGUAGE_MAPPING = {
    'de': 'de_DE',
    'ro': 'ro_RO',
    'tr': 'tr_TR',
    'zh-hans': 'zh_Hans'
}
VERSION = '3.0'


class TransifexHelper:
    def __init__(self, message_processor):
        self.message_processor = message_processor

    def load_transifex_config(self):
        transifex_config = configparser.ConfigParser()
        transifex_config_path = Path(settings.BASE_DIR, '..', '.tx', 'config')

        with transifex_config_path.open(mode='r') as file_object:
            transifex_config.read_file(file_object)

        return transifex_config

    def is_app_name_in_config(self, app_name):
        sections = self.load_transifex_config().sections()
        app_name_string = 'mayan-edms.{}'.format(app_name)
        for section in sections:
            if section.startswith(app_name_string):
                return True

        return False

    def find_missing_apps(self):
        missing_list = set()
        for app_name in self.message_processor.app_list:
            app_has_translations = getattr(
                apps.app_configs[app_name], 'has_translations', True
            )

            if app_has_translations and not self.is_app_name_in_config(app_name=app_name):
                missing_list.add(app_name)

        return missing_list

    def generate_configuration_file(self):
        # Hidden import
        from mayan.apps.dependencies import versions as version

        mayan_version = version.Version(
            version_string=self.message_processor.mayan.__version__
        )

        version_string = '{}-{}'.format(
            mayan_version.major, mayan_version.minor
        )

        output = []

        output.extend(
            ('[main]', 'host     = https://www.transifex.com')
        )

        if TRANSIFEX_LANGUAGE_MAPPING:
            language_mappings = []
            for remote_language, local_language in TRANSIFEX_LANGUAGE_MAPPING.items():
                language_mappings.append(
                    '{}: {}'.format(remote_language, local_language)
                )

            output.append(
                'lang_map = {}'.format(
                    ', '.join(language_mappings)
                )
            )
            output.append('')

        for app_label in self.message_processor.app_list:
            app = apps.get_app_config(app_label=app_label)

            has_app_translations = app.get_has_app_translations()
            has_javascript_translations = app.get_has_javascript_translations()

            resource_name_base = 'v{}.{} {}'.format(
                mayan_version.major, mayan_version.minor, app.verbose_name
            )

            if has_app_translations:
                output.append(
                    '[o:rosarior:p:mayan-edms:r:{}-{}]'.format(
                        app_label, version_string
                    )
                )
                output.append(
                    'file_filter   = {}/locale/<lang>/LC_MESSAGES/django.po'.format(
                        app.mayan_path_relative
                    )
                )
                output.append(
                    'source_file   = {}/locale/en/LC_MESSAGES/django.po'.format(
                        app.mayan_path_relative
                    )
                )
                output.append('source_lang   = en')
                output.append(
                    'resource_name = {}'.format(resource_name_base)
                )
                output.append('type          = PO')
                output.append('')

            if has_javascript_translations:
                output.append(
                    '[o:rosarior:p:mayan-edms:r:{}-{}-javascript]'.format(
                        app_label, version_string
                    )
                )
                output.append(
                    'file_filter   = {}/locale/<lang>/LC_MESSAGES/djangojs.po'.format(
                        app.mayan_path_relative
                    )
                )
                output.append(
                    'source_file   = {}/locale/en/LC_MESSAGES/djangojs.po'.format(
                        app.mayan_path_relative
                    )
                )
                output.append('source_lang = en')
                output.append(
                    'resource_name = {} (JavaScript)'.format(
                        resource_name_base
                    )
                )
                output.append('type          = PO')
                output.append('')

        print(
            '\n'.join(
                output[:-1]
            )
        )


class MessageProcessor:
    @staticmethod
    def get_app_list():
        # Hidden import.
        from mayan.apps.common.apps import MayanAppConfig

        return sorted(
            app_label for app_label, app_config in apps.app_configs.items() if issubclass(
                type(app_config), MayanAppConfig
            )
        )

    @staticmethod
    def get_language_list():
        result = []

        for language in dict(settings.LANGUAGES).keys():
            language = PROJECT_LANGUAGE_MAPPING.get(language, language)
            if '-' in language:
                elements = language.split('-')
                language = '{}_{}' .format(
                    elements[0], elements[1].upper()
                )

            result.append(language)

        return result

    def __init__(self):
        self._initialize_mayan()

        self.argument_method_map = {
            'django_compile': 'do_compilemessages',
            'django_make': 'do_makemessages',
            'django_make_javascript': 'do_makemessages_javascript',
            'transifex_missing_apps': 'do_transifex_check_missing_apps',
            'transifex_generate_config': 'do_transifex_generate_configuration_file',
            'transifex_pull': 'do_transifex_pull_translations',
            'transifex_push': 'do_transifex_push_translations'
        }

        self.parser = optparse.OptionParser(
            usage='%prog <command: {}> [options]'.format(
                ', '.join(
                    self.argument_method_map.keys()
                )
            ), version='%prog {}'.format(VERSION)
        )
        self.parser.add_option(
            '-a', '--app', help='specify which app to process', dest='app',
            action='store', metavar='appname'
        )
        self.parser.add_option(
            '-l', '--lang', help='specify which language to process',
            dest='lang', action='store', metavar='language'
        )
        (options, args) = self.parser.parse_args()

        if len(args) != 1:
            self.parser.error('command argument is missing')

        django.setup()

        self.app_list = MessageProcessor.get_app_list()

        if options.app:
            self.selected_apps = options.app.split(',')
        else:
            self.selected_apps = self.app_list

        if options.lang:
            self.selected_languages = options.lang.split(',')
        else:
            self.selected_languages = MessageProcessor.get_language_list()

        command_manage = sh.Command('django-admin')
        self.command_makemessages = command_manage.bake('makemessages')
        self.command_compilemessages = command_manage.bake('compilemessages')

        command_transifex_client = sh.Command('tx')
        self.command_transifex_pull_translations = command_transifex_client.bake('pull')
        self.command_transifex_push_translations = command_transifex_client.bake('push')

        self.call_command(
            command=args[0]
        )

    def _initialize_mayan(self):
        path_current = Path('.')
        sys.path.insert(
            1, str(
                path_current.absolute()
            )
        )
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mayan.settings')

        import mayan

        self.mayan = mayan

    def call_command(self, command):
        try:
            method = self.argument_method_map[command]
        except KeyError:
            self.parser.error(
                'Unknown command "{}"'.format(command)
            )

        return getattr(self, method)()

    def do_compilemessages(self):
        print('Compiling messages')

        for app_label in self.selected_apps:
            print('Processing app: %s...' % app_label)

            app = apps.get_app_config(app_label=app_label)

            os.chdir(app.mayan_path)

            path_locales = app.mayan_path / 'locale'
            if path_locales.exists():
                print(
                    'Doing languages: {}'.format(
                        ', '.join(self.selected_languages)
                    )
                )

                environment = os.environ.copy()
                environment.pop('DJANGO_SETTINGS_MODULE')

                command_arguments = self.get_django_language_arguments()

                self.command_compilemessages(
                    _env=environment, *command_arguments
                )
            else:
                print(
                    'Skipping app: {}, missing `locale` folder'.format(app)
                )

    def do_makemessages(self):
        print('Making messages')

        for app_label in self.selected_apps:
            print('Processing app: %s...' % app_label)

            app = apps.get_app_config(app_label=app_label)

            os.chdir(app.mayan_path)

            path_locale = app.mayan_path / 'locale'
            if path_locale.exists():
                print(
                    'Doing languages: {}'.format(
                        ', '.join(self.selected_languages)
                    )
                )

                environment = os.environ.copy()
                environment.pop('DJANGO_SETTINGS_MODULE')

                command_arguments = self.get_django_language_arguments()
                self.command_makemessages(
                    _env=environment, *command_arguments, no_obsolete=True
                )
            else:
                print(
                    'Skipping app: {}, missing `locale` folder'.format(app)
                )

    def do_makemessages_javascript(self):
        print('Making messages, Javascript')

        for app_label in self.selected_apps:
            print('Processing app: %s...' % app_label)

            app = apps.get_app_config(app_label=app_label)

            os.chdir(app.mayan_path)

            path_locale = app.mayan_path / 'locale'
            if path_locale.exists():
                print(
                    'Doing languages: {}'.format(
                        ', '.join(self.selected_languages)
                    )
                )

                environment = os.environ.copy()
                environment.pop('DJANGO_SETTINGS_MODULE')

                command_arguments = self.get_django_language_arguments()
                self.command_makemessages(
                    _env=environment, *command_arguments, domain='djangojs',
                    extension='js,html', ignore=[
                        '*/node_modules/*', '**/*.min.js', '*/vendor/*'
                    ], no_obsolete=True
                )
            else:
                print(
                    'Skipping app: {}, missing `locale` folder'.format(app)
                )

    def do_transifex_check_missing_apps(self):
        print('Checking configuration file')
        transifex_helper = TransifexHelper(message_processor=self)

        transifex_helper.find_missing_apps()

        missing_list = transifex_helper.find_missing_apps()
        if missing_list:
            print(
                'Apps without a transifex entry: {}'.format(
                    ', '.join(missing_list)
                )
            )
            exit(1)
        else:
            print('No missing apps')

    def do_transifex_generate_configuration_file(self):
        transifex_helper = TransifexHelper(message_processor=self)

        transifex_helper.generate_configuration_file()

    def do_transifex_pull_translations(self):
        print('Pulling translations')

        for app_label in self.selected_apps:
            print('Processing app: %s...' % app_label)
            print(
                'Doing languages: {}'.format(
                    ', '.join(self.selected_languages)
                )
            )
            self.command_transifex_pull_translations(
                '-f', '-l', ','.join(self.selected_languages)
            )

    def do_transifex_push_translations(self):
        print('Push translations')

        for app_label in self.selected_apps:
            print('Processing app: %s...' % app_label)
            print(
                'Doing languages: {}'.format(
                    ', '.join(self.selected_languages)
                )
            )
            self.command_transifex_push_translations(
                '-s', '-l', ','.join(self.selected_languages)
            )

    def get_django_language_arguments(self):
        command_argument_locales = []
        for language in self.selected_languages:
            command_argument_locales.append('--locale')
            command_argument_locales.append(language)

        return command_argument_locales


if __name__ == '__main__':
    message_processor = MessageProcessor()
