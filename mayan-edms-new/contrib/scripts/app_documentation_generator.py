#!/usr/bin/env python

import os
from pathlib import Path
import shutil
import sys

import django
from django.apps import apps

sys.path.insert(
    0, os.path.abspath('..')
)
sys.path.insert(
    1, os.path.abspath('.')
)

from mayan.settings import BASE_DIR


if __name__ == '__main__':
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mayan.settings')
    django.setup()

    path_documentation = Path(BASE_DIR, '..', 'docs')
    path_file_documentation_features = path_documentation / 'chapters' / 'features.txt'
    path_documentation_apps = path_documentation / 'chapters' / 'apps'

    shutil.rmtree(path=path_documentation_apps, ignore_errors=True)

    with path_file_documentation_features.open(mode='w') as file_object:
        file_object.writelines(
            ('.. _features:\n\n', '========\n', 'Features\n', '========\n', '\n')
        )

    app_list = []
    app_feature_footer_list = []

    for app_config in apps.get_app_configs():
        path_app_documentation = Path(app_config.path, 'docs')

        path_documentation_for_app = path_documentation_apps / app_config.label

        if (path_app_documentation / 'index.txt').exists():
            app_list.append(app_config.label)
            path_documentation_for_app.mkdir(exist_ok=True, parents=True)

            try:
                shutil.copytree(
                    dirs_exist_ok=True, dst=path_documentation_for_app,
                    ignore=shutil.ignore_patterns('_*.txt',),
                    src=path_app_documentation
                )
            except FileNotFoundError:
                """
                Non fatal, just means the app does not provide documentation.
                """

        try:
            with (path_app_documentation / '_features.txt').open(mode='r') as file_object_source:
                with (path_documentation / 'chapters' / 'features.txt').open(mode='a') as file_object_target:
                    shutil.copyfileobj(
                        fsrc=file_object_source, fdst=file_object_target
                    )

            try:
                with (path_app_documentation / '_features_footer.txt').open(mode='r') as file_object_source:
                    app_feature_footer_list.append(
                        file_object_source.read()
                    )
            except FileNotFoundError:
                """
                Non fatal, just means the app does not provide features footer.
                """
        except FileNotFoundError:
            """
            Non fatal, just means the app does not provide features.
            """

    with (path_documentation / 'chapters' / 'features.txt').open(mode='a') as file_object:
        file_object.writelines(app_feature_footer_list)

    app_list.sort()

    with (path_documentation_apps / 'index.txt').open(mode='w') as file_object:
        title = 'Apps documentation'
        title_length = len(title)

        file_object.writelines(
            (
                '{} \n'.format(title), '{}\n'.format('=' * title_length), '\n'
            )
        )

        for app in app_list:
            file_object.write(
                '- :doc:`{}/index`\n'.format(app)
            )

        file_object.write('\n.. toctree::\n')
        file_object.write('    :hidden:\n')
        file_object.write('\n')

        for app in app_list:
            file_object.write(
                '    {}/index\n'.format(app)
            )
