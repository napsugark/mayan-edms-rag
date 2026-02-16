#!/usr/bin/env python

from datetime import datetime
import os
from pathlib import Path
import sys

from dateutil import parser
import sh

import django
from django.template import Template, Context
from django.utils.encoding import force_str

path_current = Path('.')

sys.path.append(
    str(
        path_current.resolve()
    )
)

import mayan  # NOQA
from mayan.settings import BASE_DIR  # NOQA
from mayan.settings import literals  # NOQA

FILENAME_REQUIREMENTS = 'requirements.txt'
FILENAME_SETUP = 'setup.py'
FILENAME_TEMPLATE_MAYAN_INIT = '__init__.py.tmpl'
FILENAME_TEMPLATE_LICENSE = 'LICENSE.tmpl'
FILENAME_TEMPLATE_SETUP = 'setup.py.tmpl'
LIST_FILENAME_LICENSE = ('LICENSE', 'mayan/LICENSE')


class SetupUpdater:
    def __init__(self):
        self.path_base = Path(BASE_DIR, '..')

        try:
            self.git_build = sh.Command('git').bake('describe', '--tags', '--always', 'HEAD')
            self.git_date = sh.Command('git').bake('--no-pager', 'log', '-1', '--format=%cd')
        except sh.CommandNotFound:
            self.git_build = None
            self.git_date = None

    def do_django_setup(self):
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mayan.settings')
        django.setup()

    def do_execute(self):
        self.do_django_setup()
        self.do_license_update()
        self.do_init_update()
        self.do_setup_update()

    def do_init_update(self):
        path_mayan_init = self.path_base / 'mayan' / '__init__.py'

        upstream_build = '0x{:06X}'.format(mayan.__build__)
        year_copyright = self.get_copyright_year()
        build_string = self.get_build_number()
        timestamp = self.get_commit_timestamp()
        version = self.get_version()

        context = {
            'build': upstream_build,
            'build_string': build_string,
            'django_series': literals.DJANGO_SERIES,
            'timestamp': timestamp,
            'version': version,
            'year_copyright': year_copyright
        }

        self.do_template_render(
            dict_context=context, path_destination=path_mayan_init,
            path_source=FILENAME_TEMPLATE_MAYAN_INIT
        )

    def do_license_update(self):
        year_copyright = self.get_copyright_year()
        context = {'year_copyright': year_copyright}

        for license_filename in LIST_FILENAME_LICENSE:
            self.do_template_render(
                dict_context=context, path_destination=license_filename,
                path_source=FILENAME_TEMPLATE_LICENSE
            )

    def do_setup_update(self):
        requirements = self.get_requirements(
            directory=self.path_base, filename=FILENAME_REQUIREMENTS
        )
        context = {'requirements': requirements}

        self.do_template_render(
            dict_context=context, path_destination=FILENAME_SETUP,
            path_source=FILENAME_TEMPLATE_SETUP
        )

    def do_template_render(self, dict_context, path_source, path_destination):
        with Path(path_source).open() as file_object:
            template = file_object.read()
            context = Context(dict_context)
            result = Template(template).render(context=context)

            with Path(path_destination).open(mode='w') as file_object:
                file_object.write(result)

    def get_build_number(self):
        if self.git_build and self.git_date:
            try:
                result = '{}_{}'.format(
                    self.git_build(), self.git_date()
                ).replace('\n', '')
            except sh.ErrorReturnCode_128:
                result = ''
        else:
            result = ''
        return result

    def get_commit_timestamp(self):
        datetime = parser.parse(
            force_str(
                s=self.git_date()
            )
        )
        return datetime.strftime('%y%m%d%H%M')

    def get_copyright_year(self):
        now = datetime.now()
        return now.year

    def get_requirements(self, directory, filename):
        result_final = []

        path_requirements = Path(directory, filename)

        with path_requirements.open() as file_object:
            for line in file_object:
                if line.startswith('-r'):
                    line = line.split('\n')[0][3:]
                    path_child = Path(line)

                    directory = path_child.parent
                    filename = path_child.name

                    path_directory = self.path_base / directory

                    result = self.get_requirements(
                        directory=path_directory, filename=filename
                    )

                    result_final.extend(result)
                elif not line.startswith('\n'):
                    result_final.append(
                        line.split('\n')[0]
                    )

        return result_final

    def get_version(self):
        # Ignore local version if any.
        version_upstream = '.'.join(
            mayan.__version__.split('+')[0].split('.')
        )

        version_local = getattr(literals, 'LOCAL_VERSION')

        if version_local:
            version_final = '{}+{}'.format(version_upstream, version_local)
        else:
            version_final = version_upstream

        return version_final


if __name__ == '__main__':
    app = SetupUpdater()
    app.do_execute()
