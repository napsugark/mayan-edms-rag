#!/usr/bin/env python

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import mayan

PACKAGE_NAME = 'mayan-edms'
PACKAGE_DIR = 'mayan'

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()


def fullsplit(path, result=None):
    """
    Split a pathname into components (the opposite of os.path.join) in a
    platform-neutral way.
    """
    if result is None:
        result = []
    head, tail = os.path.split(path)
    if head == '':
        return [tail] + result
    if head == path:
        return result
    return fullsplit(head, [tail] + result)


def find_packages(directory):
    # Compile the list of packages available, because distutils doesn't have
    # an easy way to do this.
    packages, data_files = [], []
    root_dir = os.path.dirname(__file__)
    if root_dir != '':
        os.chdir(root_dir)

    for dirpath, dirnames, filenames in os.walk(directory):
        if not dirpath.startswith('mayan/media'):
            # Ignore dirnames that start with '.'
            if os.path.basename(dirpath).startswith('.'):
                continue
            if '__init__.py' in filenames:
                packages.append(
                    '.'.join(
                        fullsplit(dirpath)
                    )
                )
            elif filenames:
                data_files.append(
                    [
                        dirpath, [
                            os.path.join(dirpath, filename) for filename in filenames
                        ]
                    ]
                )

    return packages


install_requires = """
django==4.2.24
CairoSVG==2.8.2
Markdown==3.9
Pillow==11.3.0
PyYAML==6.0.2
Whoosh==2.7.4
boto3==1.40.24
celery==5.5.3
dateparser==1.2.2
django-activity-stream==2.0.0
django-auth-ldap==5.1.0
django-celery-beat==2.8.1
django-cors-headers==4.7.0
django-formtools==2.3
django-model-utils==5.0.0
django-mptt==0.18.0
django-qsstats-magic==1.1.0
django-solo==2.4.0
django-storages==1.14.6
django-stronghold==0.4.0
django-widget-tweaks==1.5.0
djangorestframework==3.16.1
djangorestframework-recursive==0.1.2
drf-yasg==1.21.10
elasticsearch==9.1.0
extract-msg==0.55.0
flex==6.14.1
furl==2.1.4
fusepy==3.0.1
gevent==25.8.2
google-cloud-storage==3.3.1
graphviz==0.21
greenlet==3.2.4
gunicorn==23.0.0
jsonschema==4.25.1
mozilla-django-oidc==4.0.1
nh3==0.3.0
node-semver==0.9.0
ollama==0.5.3
openai==1.107.2
pycountry==24.6.1
pycryptodome==3.23.0
pyotp==2.9.0
pypdf==6.0.0
python-dateutil==2.9.0.post0
python_gnupg==0.5.5
pytz==2025.2
qrcode==8.2
requests==2.32.5
sentry-sdk==2.37.0
sh==2.2.2
swagger-spec-validator==3.0.4
whitenoise==6.9.0
""".split()

with open(file='README.rst') as file_object:
    readme = file_object.read()

setup(
    author='Roberto Rosario',
    author_email='roberto.rosario@mayan-edms.com',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Web Environment',
        'Framework :: Django',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Internet :: WWW/HTTP',
        'Topic :: Internet :: WWW/HTTP :: WSGI :: Application',
        'Topic :: Communications :: File Sharing'
    ],
    description=mayan.__description__,
    include_package_data=True,
    install_requires=install_requires,
    license='"GPL-2.0-only',
    long_description=readme,
    name=PACKAGE_NAME,
    packages=find_packages(PACKAGE_DIR),
    platforms=['any'],
    project_urls={
        'Documentation': 'https://docs.mayan-edms.com/',
        'Forum': 'https://forum.mayan-edms.com/',
        'Changelog': 'https://gitlab.com/mayan-edms/mayan-edms/-/blob/master/HISTORY.rst',
        'Bug Tracker': 'https://gitlab.com/mayan-edms/mayan-edms/-/issues',
        'Source Code': 'https://gitlab.com/mayan-edms/mayan-edms',
        'Support': 'https://www.mayan-edms.com/support/'
    },
    python_requires='>=3.9',
    scripts=['mayan/bin/mayan-edms.py'],
    url=mayan.__website__,
    version=mayan.__version__,
    zip_safe=False
)
