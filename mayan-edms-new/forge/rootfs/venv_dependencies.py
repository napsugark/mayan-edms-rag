#!/usr/bin/env python

import os
import subprocess
import sys

dependency_string = os.environ.get('FORGE_PYTHON_PACKAGES', '')
dependency_list = dependency_string.split(' ')

for dependency in dependency_list:
    print(
        'Installing base dependency: {dependency}'.format(dependency=dependency),
        flush=True
    )
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--upgrade', dependency],
        check=True
    )
