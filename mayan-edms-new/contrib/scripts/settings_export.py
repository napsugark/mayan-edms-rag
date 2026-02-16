#!/usr/bin/env python

import optparse
import os
from pathlib import Path
import sys

import markdown

import django

VERSION = '1.0'


class SettingNamespaceExporter:
    def __init__(self, name, output_format):
        self.output_format = output_format
        self.name = name

        self._initialize_mayan()

        django.setup()

        # Hidden imports
        from mayan.apps.smart_settings.setting_clusters import SettingCluster
        from mayan.apps.smart_settings.settings import setting_cluster

        SettingCluster.load_modules()

        try:
            self.setting_namespace = setting_cluster.get_namespace(
                name=self.name
            )
        except KeyError as exception:
            raise KeyError(
                'Unknown settings namespace `{}`.'.format(self.name)
            ) from exception

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

    def do_export(self):
        output = []

        output.append(
            '# {}'.format(self.setting_namespace.label)
        )

        for setting in self.setting_namespace.get_setting_list():
            output_setting = self.do_setting_render(setting=setting)

            output.extend(output_setting)

        self.output = '\n'.join(output)

        return self.get_as_markdown()

    def get_as_html(self):
        return markdown.markdown(self.output)

    def get_as_markdown(self):
        return self.output

    def do_setting_render(self, setting):
        output = []

        output.append('## `{}`'.format(setting.global_name))
        output.append(str(setting.help_text))
        output.append('Default: `{}`'.format(setting.default))

        if setting.choices:
            output.append('Choices: {}'.format(setting.choices))

        return output


if __name__ == '__main__':
    parser = optparse.OptionParser(
        usage='%prog [version number]', version='%prog {}'.format(VERSION)
    )
    parser.add_option(
        '-f', '--format', help='specify the output format',
        dest='output_format',
        action='store', metavar='output_format'
    )

    (options, args) = parser.parse_args()

    if len(args) != 1:
        parser.error('Setting namespace name argument is missing')

    setting_namespace_name = args[0]

    app = SettingNamespaceExporter(
        name=setting_namespace_name,
        output_format=options.output_format,
    )
    result = app.do_export()
    print(result)
