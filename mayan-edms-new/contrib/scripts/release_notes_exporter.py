#!/usr/bin/env python

import optparse
from pathlib import Path

from docutils import core
from lxml import etree, html
import sh

MONTHS_TO_NUMBER = {
    'January': 1,
    'February': 2,
    'March': 3,
    'April': 4,
    'May': 5,
    'June': 6,
    'July': 7,
    'August': 8,
    'September': 9,
    'October': 10,
    'November': 11,
    'December': 12
}
VERSION = '3.0'
ignore_ids_list = ('troubleshooting', 'upgrade-process', 'upgrading-process')


class ReleaseNoteExporter:
    @staticmethod
    def filter_elements(tree):
        result = []

        for element in tree:
            if element.attrib.get('id') not in ignore_ids_list:
                if element.tag == 'div':
                    if element.attrib.get('id') not in ignore_ids_list:
                        result.extend(
                            ReleaseNoteExporter.filter_elements(tree=element)
                        )
                else:
                    if element.attrib.get('id') not in ignore_ids_list:
                        result.append(
                            etree.tostring(element).replace(b'\n', b' ')
                        )

        return result

    def __init__(self, output_format, version, releases_path):
        self.releases_path = releases_path or Path('.').resolve() / 'docs' / 'chapters' / 'releases'
        self.output_format = output_format
        self.version = version

    def export(self):
        path_documentation = Path(self.releases_path) / '{}.txt'.format(self.version)

        with path_documentation.open(mode='r') as file_object:
            content = []

            for line in file_object:
                if line.startswith('.. include'):
                    # Skip
                    pass
                elif ':gitlab-issue:' in line:
                    line_parts = line.split('`')

                    result = (
                        '- `GitLab issue #{} '
                        '<https://gitlab.com/mayan-edms/mayan-edms/issues/{}>`_ {}'.format(
                            line_parts[1], line_parts[1], line_parts[2]
                        )
                    )

                    content.append(result)
                elif ':github-issue:' in line:
                    line_parts = line.split('`')

                    result = (
                        '- `GitHub issue #{} '
                        '<https://github.com/mayan-edms/mayan-edms/issues/{}>`_ {}'.format(
                            line_parts[1], line_parts[1], line_parts[2]
                        )
                    )

                    content.append(result)
                else:
                    content.append(line)

        parts = core.publish_parts(
            source=''.join(content), writer_name='html'
        )
        html_fragment = '{}{}'.format(
            parts['body_pre_docinfo'], parts['fragment']
        )

        result = ReleaseNoteExporter.filter_elements(
            tree=html.fromstring(html_fragment)
        )

        if self.output_format == 'md':
            command_pandoc = sh.Command('pandoc')

            markdown_tag_cleanup = (
                (b'class="docutils literal"', b''),
                (b'class="reference external"', b''),
            )

            joined_result = b''.join(result)

            for markdown_tag_cleanup_item in markdown_tag_cleanup:
                joined_result = joined_result.replace(*markdown_tag_cleanup_item)

            return command_pandoc(_in=joined_result, f='html', t='markdown')
        elif self.output_format == 'news':
            command_pandoc = sh.Command('pandoc')

            markdown_tag_cleanup = (
                (b'class="docutils literal"', b''),
                (b'class="reference external"', b''),
            )

            joined_result = b''.join(result)

            for markdown_tag_cleanup_item in markdown_tag_cleanup:
                joined_result = joined_result.replace(*markdown_tag_cleanup_item)

            result_body = command_pandoc(_in=joined_result, f='html', t='markdown')

            tree = html.fromstring(html_fragment)

            released, month, day, year = tree[1].text.split(' ')

            return '\n'.join(
                (
                    '---',
                    'date: {}-{:02d}-{:02d}'.format(
                        year, MONTHS_TO_NUMBER[month], int(
                            day[:-1]
                        )
                    ),
                    'title: "{}"'.format(
                        tree[0].text
                    ),
                    '---',
                    str(result_body)
                )
            )
        else:
            return b''.join(result)


if __name__ == '__main__':
    parser = optparse.OptionParser(
        usage='%prog [version number]', version='%prog {}'.format(VERSION)
    )
    parser.add_option(
        '-f', '--format', help='specify the output format',
        dest='output_format',
        action='store', metavar='output_format'
    )
    parser.add_option(
        '-p', '--releases-path', help='path to the releases directory',
        dest='releases_path',
        action='store', metavar='releases_path'
    )

    (options, args) = parser.parse_args()

    if len(args) != 1:
        parser.error('version argument is missing')

    version = args[0]

    message_processor = ReleaseNoteExporter(
        output_format=options.output_format,
        releases_path=options.releases_path, version=args[0]
    )
    result = message_processor.export()
    print(result)
