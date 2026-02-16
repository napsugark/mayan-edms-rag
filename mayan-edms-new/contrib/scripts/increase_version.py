#!/usr/bin/env python
import os
import sys


if __name__ == '__main__':
    sys.path.insert(
        1, os.path.abspath('.')
    )
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mayan.settings')
    from mayan.apps.dependencies import versions as version

    if len(sys.argv) == 1:
        print(
            'usage: <part to increase [major, minor, micro, dev, pre, post]>'
        )
        exit(0)

    if len(sys.argv) < 3:
        print('Insufficient arguments')
        exit(1)

    version_string = sys.argv[1]
    if version_string == '-':
        version_string = sys.stdin.read().replace('\n', '')

    version = version.Version(version_string=version_string)
    part = sys.argv[2].lower()

    if part == 'major':
        version.increment_major()
    elif part == 'minor':
        version.increment_minor()
    elif part == 'micro':
        version.increment_micro()
    elif part == 'dev':
        version.increment_dev()
    elif part == 'pre':
        version.increment_pre()
    elif part == 'post':
        version.increment_post()
    else:
        print('Unknown part')
        exit(1)

    print(
        version.get_version_string()
    )
