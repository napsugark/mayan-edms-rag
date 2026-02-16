from django.template import Library

import mayan
from mayan.apps.dependencies.versions import Version
from mayan.apps.platforms.utils import yaml_dump
from mayan.settings.literals import LINUX_PACKAGES_DEBIAN_PUSH

register = Library()


@register.simple_tag(name='platforms_gitlab_ci_cache_before_script')
def tag_platforms_gitlab_ci_cache_before_script(indent, apk=False, apt=False):
    data = []

    if apk:
        data.extend(
            [
                'mkdir --parents ${APK_CACHE_DIR}',
                'apk update'
            ]
        )

    if apt:
        data.extend(
            [
                'export APT_STATE_LISTS=${APT_CACHE_DIR}/lists && export APT_CACHE_ARCHIVES=${APT_CACHE_DIR}/archives',
                'mkdir --parents "${APT_STATE_LISTS}/partial" && mkdir --parents "${APT_CACHE_ARCHIVES}/partial"',
                'printf "dir::state::lists    ${APT_STATE_LISTS};\\ndir::cache::archives    ${APT_CACHE_ARCHIVES};\\n" > /etc/apt/apt.conf.d/99gitlab-ci-cache',
                'if [ "${APT_PROXY}" ]; then echo "Acquire::http { Proxy \\"http://${APT_PROXY}\\"; };" > /etc/apt/apt.conf.d/01proxy; fi',
                'apt-get update'
            ]
        )

    return yaml_dump(data=data, indent=indent)


@register.simple_tag(name='platforms_gitlab_ci_cache_paths')
def tag_platforms_gitlab_ci_cache_paths(
    indent, apk=False, apt=False, pip=False
):
    cache_list = []

    version_base = Version(version_string=mayan.__version__)
    version_upstream = Version(
        version_string=version_base.as_upstream()
    )
    version_final = version_upstream.as_minor()

    if apk:
        cache_list.append(
            {
                'key': 'apk-cache-{}'.format(version_final),
                'paths': ['${APK_CACHE_DIR}']
            }
        )

    if apt:
        cache_list.append(
            {
                'key': 'apt-cache-{}'.format(version_final),
                'paths': ['${APT_CACHE_DIR}']
            }
        )

    if pip:
        cache_list.append(
            {
                'key': 'pip-cache-{}'.format(version_final),
                'paths': ['${PIP_CACHE_DIR}']
            }
        )

    return yaml_dump(data=cache_list, indent=indent)


@register.simple_tag(name='platforms_gitlab_ci_cache_variables')
def tag_platforms_gitlab_ci_cache_variables(
    indent, apk=False, apt=False, pip=False
):
    variables = {}

    if apk:
        variables['APK_CACHE_DIR'] = '${CI_PROJECT_DIR}/.cache/apk'

    if apt:
        variables['APT_CACHE_DIR'] = '${CI_PROJECT_DIR}/.cache/apt'

    if pip:
        variables['PIP_CACHE_DIR'] = '${CI_PROJECT_DIR}/.cache/pip'

    return yaml_dump(data=variables, indent=indent)


@register.simple_tag(name='platforms_gitlab_ci_config_env_before_script')
def tag_platforms_gitlab_ci_config_env_before_script(indent):
    data = [
        r'set -a && sed -E "s/=(.*)/\=\"\1\"/g" config.env > /tmp/config.env && . /tmp/config.env && rm /tmp/config.env && set +a'
    ]

    return yaml_dump(data=data, indent=indent)


@register.simple_tag(name='platforms_gitlab_ci_ssh_before_script')
def tag_platforms_gitlab_ci_ssh_before_script(indent, hostname, private_key):
    data = [
        'mkdir --parents ~/.ssh',
        'chmod 700 ~/.ssh',
        'echo "{}" > ~/.ssh/known_hosts'.format(hostname),
        'chmod 644 ~/.ssh/known_hosts',
        '\'which ssh-agent || ( apt-get update --yes && apt-get install --yes --no-install-recommends {debian_packages} )\''.format(debian_packages=LINUX_PACKAGES_DEBIAN_PUSH),
        'eval $(ssh-agent -s)',
        'echo "{}" | tr -d \'\\r\' | ssh-add - > /dev/null'.format(private_key)
    ]

    return yaml_dump(data=data, indent=indent)
