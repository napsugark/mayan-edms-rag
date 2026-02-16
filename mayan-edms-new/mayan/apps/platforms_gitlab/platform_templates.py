from django.utils.translation import gettext_lazy as _

from mayan.apps.platforms.platform_templates import PlatformTemplate, Variable
from mayan.settings.literals import (
    DEFAULT_DATABASE_NAME, DEFAULT_DATABASE_PASSWORD, DEFAULT_DATABASE_USER,
    DOCKER_CLI_IMAGE_VERSION, DOCKER_DIND_IMAGE_VERSION,
    DOCKER_LINUX_IMAGE_NAME, DOCKER_LINUX_IMAGE_TAG,
    DOCKER_MYSQL_IMAGE_NAME, DOCKER_MYSQL_IMAGE_TAG,
    DOCKER_POSTGRESQL_IMAGE_NAME, DOCKER_POSTGRESQL_IMAGE_TAG,
    GITLAB_CI_BRANCH_BUILDS_DOCKER, GITLAB_CI_BRANCH_BUILDS_DOCUMENTATION,
    GITLAB_CI_BRANCH_BUILDS_PYTHON, GITLAB_CI_BRANCH_DEPLOYMENTS_DEMO,
    GITLAB_CI_BRANCH_DEPLOYMENTS_STAGING, GITLAB_CI_BRANCH_RELEASES_ALL_MAJOR,
    GITLAB_CI_BRANCH_RELEASES_ALL_MINOR,
    GITLAB_CI_BRANCH_RELEASES_DOCKER_MAJOR,
    GITLAB_CI_BRANCH_RELEASES_DOCKER_MINOR,
    GITLAB_CI_BRANCH_RELEASES_DOCUMENTATION,
    GITLAB_CI_BRANCH_RELEASES_NIGHTLY, GITLAB_CI_BRANCH_RELEASES_PYTHON_MAJOR,
    GITLAB_CI_BRANCH_RELEASES_PYTHON_MINOR, GITLAB_CI_BRANCH_RELEASES_STAGING,
    GITLAB_CI_BRANCH_RELEASES_TESTING, GITLAB_CI_BRANCH_TESTS_ALL,
    GITLAB_CI_BRANCH_TESTS_DOCKER, GITLAB_CI_BRANCH_TESTS_PYTHON_ALL,
    GITLAB_CI_BRANCH_TESTS_PYTHON_BASE, GITLAB_CI_BRANCH_TESTS_PYTHON_UPGRADE,
    GITLAB_CI_BRANCH_TRY_STAGING, LINUX_PACKAGES_ALPINE_BUILD,
    LINUX_PACKAGES_ALPINE_PYTHON, LINUX_PACKAGES_DEBIAN_BASE,
    LINUX_PACKAGES_DEBIAN_BUILD, LINUX_PACKAGES_DEBIAN_DOCUMENTATION,
    LINUX_PACKAGES_DEBIAN_POSTGRESQL, LINUX_PACKAGES_DEBIAN_PYTHON,
    LINUX_PACKAGES_DEBIAN_TEST
)


class PlatformTemplateGitLabCI(PlatformTemplate):
    label = _(message='Template that generates a GitLab CI config file.')
    name = 'gitlab-ci'
    template_name = 'platform/gitlab/gitlab-ci.tmpl'

    def __init__(self):
        self.variables = (
            Variable(
                name='DEFAULT_DATABASE_NAME',
                default=DEFAULT_DATABASE_NAME,
                environment_name='MAYAN_DEFAULT_DATABASE_NAME'
            ),
            Variable(
                name='DEFAULT_DATABASE_PASSWORD',
                default=DEFAULT_DATABASE_PASSWORD,
                environment_name='MAYAN_DEFAULT_DATABASE_PASSWORD'
            ),
            Variable(
                name='DEFAULT_DATABASE_USER',
                default=DEFAULT_DATABASE_USER,
                environment_name='MAYAN_DEFAULT_DATABASE_USER'
            ),
            Variable(
                name='DOCKER_CLI_IMAGE_VERSION',
                default=DOCKER_CLI_IMAGE_VERSION,
                environment_name='MAYAN_DOCKER_CLI_IMAGE_VERSION'
            ),
            Variable(
                name='DOCKER_DIND_IMAGE_VERSION',
                default=DOCKER_DIND_IMAGE_VERSION,
                environment_name='MAYAN_DOCKER_DIND_IMAGE_VERSION'
            ),
            Variable(
                name='DOCKER_LINUX_IMAGE_NAME',
                default=DOCKER_LINUX_IMAGE_NAME,
                environment_name='MAYAN_DOCKER_LINUX_IMAGE_NAME'
            ),
            Variable(
                name='DOCKER_LINUX_IMAGE_TAG',
                default=DOCKER_LINUX_IMAGE_TAG,
                environment_name='MAYAN_DOCKER_LINUX_IMAGE_TAG'
            ),
            Variable(
                name='DOCKER_MYSQL_IMAGE_NAME',
                default=DOCKER_MYSQL_IMAGE_NAME,
                environment_name='MAYAN_DOCKER_MYSQL_IMAGE_NAME'
            ),
            Variable(
                name='DOCKER_MYSQL_IMAGE_TAG',
                default=DOCKER_MYSQL_IMAGE_TAG,
                environment_name='MAYAN_DOCKER_MYSQL_IMAGE_TAG'
            ),
            Variable(
                name='DOCKER_POSTGRESQL_IMAGE_NAME',
                default=DOCKER_POSTGRESQL_IMAGE_NAME,
                environment_name='MAYAN_DOCKER_POSTGRESQL_IMAGE_NAME'
            ),
            Variable(
                name='DOCKER_POSTGRESQL_IMAGE_TAG',
                default=DOCKER_POSTGRESQL_IMAGE_TAG,
                environment_name='MAYAN_DOCKER_POSTGRESQL_IMAGE_TAG'
            ),
            Variable(
                name='GITLAB_CI_BRANCH_BUILDS_DOCKER',
                default=GITLAB_CI_BRANCH_BUILDS_DOCKER,
                environment_name='MAYAN_GITLAB_CI_BRANCH_BUILDS_DOCKER'
            ),
            Variable(
                name='GITLAB_CI_BRANCH_BUILDS_DOCUMENTATION',
                default=GITLAB_CI_BRANCH_BUILDS_DOCUMENTATION,
                environment_name='MAYAN_GITLAB_CI_BRANCH_BUILDS_DOCUMENTATION'
            ),
            Variable(
                name='GITLAB_CI_BRANCH_BUILDS_PYTHON',
                default=GITLAB_CI_BRANCH_BUILDS_PYTHON,
                environment_name='MAYAN_GITLAB_CI_BRANCH_BUILDS_PYTHON'
            ),
            Variable(
                name='GITLAB_CI_BRANCH_DEPLOYMENTS_DEMO',
                default=GITLAB_CI_BRANCH_DEPLOYMENTS_DEMO,
                environment_name='MAYAN_GITLAB_CI_BRANCH_DEPLOYMENTS_DEMO'
            ),
            Variable(
                name='GITLAB_CI_BRANCH_DEPLOYMENTS_STAGING',
                default=GITLAB_CI_BRANCH_DEPLOYMENTS_STAGING,
                environment_name='MAYAN_GITLAB_CI_BRANCH_DEPLOYMENTS_STAGING'
            ),
            Variable(
                name='GITLAB_CI_BRANCH_RELEASES_ALL_MAJOR',
                default=GITLAB_CI_BRANCH_RELEASES_ALL_MAJOR,
                environment_name='MAYAN_GITLAB_CI_BRANCH_RELEASES_ALL_MAJOR'
            ),
            Variable(
                name='GITLAB_CI_BRANCH_RELEASES_ALL_MINOR',
                default=GITLAB_CI_BRANCH_RELEASES_ALL_MINOR,
                environment_name='MAYAN_GITLAB_CI_BRANCH_RELEASES_ALL_MINOR'
            ),
            Variable(
                name='GITLAB_CI_BRANCH_RELEASES_DOCKER_MAJOR',
                default=GITLAB_CI_BRANCH_RELEASES_DOCKER_MAJOR,
                environment_name='MAYAN_GITLAB_CI_BRANCH_RELEASES_DOCKER_MAJOR'
            ),
            Variable(
                name='GITLAB_CI_BRANCH_RELEASES_DOCKER_MINOR',
                default=GITLAB_CI_BRANCH_RELEASES_DOCKER_MINOR,
                environment_name='MAYAN_GITLAB_CI_BRANCH_RELEASES_DOCKER_MINOR'
            ),
            Variable(
                name='GITLAB_CI_BRANCH_RELEASES_DOCUMENTATION',
                default=GITLAB_CI_BRANCH_RELEASES_DOCUMENTATION,
                environment_name='MAYAN_GITLAB_CI_BRANCH_RELEASES_DOCUMENTATION'
            ),
            Variable(
                name='GITLAB_CI_BRANCH_RELEASES_NIGHTLY',
                default=GITLAB_CI_BRANCH_RELEASES_NIGHTLY,
                environment_name='MAYAN_GITLAB_CI_BRANCH_RELEASES_NIGHTLY'
            ),
            Variable(
                name='GITLAB_CI_BRANCH_RELEASES_PYTHON_MAJOR',
                default=GITLAB_CI_BRANCH_RELEASES_PYTHON_MAJOR,
                environment_name='MAYAN_GITLAB_CI_BRANCH_RELEASES_PYTHON_MAJOR'
            ),
            Variable(
                name='GITLAB_CI_BRANCH_RELEASES_PYTHON_MINOR',
                default=GITLAB_CI_BRANCH_RELEASES_PYTHON_MINOR,
                environment_name='MAYAN_GITLAB_CI_BRANCH_RELEASES_PYTHON_MINOR'
            ),
            Variable(
                name='GITLAB_CI_BRANCH_RELEASES_STAGING',
                default=GITLAB_CI_BRANCH_RELEASES_STAGING,
                environment_name='MAYAN_GITLAB_CI_BRANCH_RELEASES_STAGING'
            ),
            Variable(
                name='GITLAB_CI_BRANCH_RELEASES_TESTING',
                default=GITLAB_CI_BRANCH_RELEASES_TESTING,
                environment_name='MAYAN_GITLAB_CI_BRANCH_RELEASES_TESTING'
            ),
            Variable(
                name='GITLAB_CI_BRANCH_TESTS_ALL',
                default=GITLAB_CI_BRANCH_TESTS_ALL,
                environment_name='MAYAN_GITLAB_CI_BRANCH_TESTS_ALL'
            ),
            Variable(
                name='GITLAB_CI_BRANCH_TESTS_DOCKER',
                default=GITLAB_CI_BRANCH_TESTS_DOCKER,
                environment_name='MAYAN_GITLAB_CI_BRANCH_TESTS_DOCKER'
            ),
            Variable(
                name='GITLAB_CI_BRANCH_TESTS_PYTHON_ALL',
                default=GITLAB_CI_BRANCH_TESTS_PYTHON_ALL,
                environment_name='MAYAN_GITLAB_CI_BRANCH_TESTS_PYTHON_ALL'
            ),
            Variable(
                name='GITLAB_CI_BRANCH_TESTS_PYTHON_BASE',
                default=GITLAB_CI_BRANCH_TESTS_PYTHON_BASE,
                environment_name='MAYAN_GITLAB_CI_BRANCH_TESTS_PYTHON_BASE'
            ),
            Variable(
                name='GITLAB_CI_BRANCH_TESTS_PYTHON_UPGRADE',
                default=GITLAB_CI_BRANCH_TESTS_PYTHON_UPGRADE,
                environment_name='MAYAN_GITLAB_CI_BRANCH_TESTS_PYTHON_UPGRADE'
            ),
            Variable(
                name='GITLAB_CI_BRANCH_TRY_STAGING',
                default=GITLAB_CI_BRANCH_TRY_STAGING,
                environment_name='MAYAN_GITLAB_CI_BRANCH_TRY_STAGING'
            )
        )

    def get_context(self):
        return {
            'LINUX_PACKAGES_ALPINE_BUILD': LINUX_PACKAGES_ALPINE_BUILD,
            'LINUX_PACKAGES_ALPINE_PYTHON': LINUX_PACKAGES_ALPINE_PYTHON,
            'LINUX_PACKAGES_DEBIAN_BASE': LINUX_PACKAGES_DEBIAN_BASE,
            'LINUX_PACKAGES_DEBIAN_BUILD': LINUX_PACKAGES_DEBIAN_BUILD,
            'LINUX_PACKAGES_DEBIAN_DOCUMENTATION': LINUX_PACKAGES_DEBIAN_DOCUMENTATION,
            'LINUX_PACKAGES_DEBIAN_POSTGRESQL': LINUX_PACKAGES_DEBIAN_POSTGRESQL,
            'LINUX_PACKAGES_DEBIAN_PYTHON': LINUX_PACKAGES_DEBIAN_PYTHON,
            'LINUX_PACKAGES_DEBIAN_TEST': LINUX_PACKAGES_DEBIAN_TEST
        }
