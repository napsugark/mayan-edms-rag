from django.utils.translation import gettext_lazy as _

from mayan.apps.platforms.platform_templates import PlatformTemplate, Variable
from mayan.settings.literals import (
    FORGE_DOCKER_BASE_IMAGE_NAME, FORGE_DOCKER_BASE_IMAGE_TAG,
    FORGE_DOCKER_COMPOSE_PROJECT_NAME, FORGE_DOCKER_IMAGE_NAME,
    FORGE_PYTHON_PACKAGES, FORGE_PYTHON_VERSION, FORGE_TRANSIFEX_VERSION,
    LINUX_PACKAGES_DEBIAN_BASE, LINUX_PACKAGES_DEBIAN_BUILD,
    LINUX_PACKAGES_DEBIAN_DOCUMENTATION, LINUX_PACKAGES_DEBIAN_FORGE,
    LINUX_PACKAGES_DEBIAN_MYSQL, LINUX_PACKAGES_DEBIAN_POSTGRESQL,
    LINUX_PACKAGES_DEBIAN_PUSH, LINUX_PACKAGES_DEBIAN_PYTHON,
    LINUX_PACKAGES_DEBIAN_TEST
)


class PlatformTemplateForgeDockerComposefile(PlatformTemplate):
    label = _(
        message='Template that generates the Mayan Forge Docker Compose file.'
    )
    name = 'forge_docker_compose'
    template_name = 'platform/forge/docker-compose.yml.tmpl'

    def __init__(self):
        with open(file='docker/rootfs/version', mode='r') as file_object:
            version_string = file_object.readline()
            version_string_clean = version_string.strip()

        self.variables = (
            Variable(
                name='FORGE_DOCKER_COMPOSE_PROJECT_NAME',
                default=FORGE_DOCKER_COMPOSE_PROJECT_NAME,
                environment_name='MAYAN_FORGE_DOCKER_COMPOSE_PROJECT_NAME'
            ),
            Variable(
                name='FORGE_DOCKER_COMPOSE_PROJECT_VERSION',
                default=version_string_clean,
                environment_name='MAYAN_FORGE_DOCKER_COMPOSE_PROJECT_VERSION'
            ),
            Variable(
                name='FORGE_DOCKER_IMAGE_NAME',
                default=FORGE_DOCKER_IMAGE_NAME,
                environment_name='MAYAN_FORGE_DOCKER_IMAGE_NAME'
            ),
            Variable(
                name='FORGE_DOCKER_IMAGE_TAG',
                default=version_string_clean,
                environment_name='MAYAN_FORGE_DOCKER_IMAGE_TAG'
            )
        )


class PlatformTemplateForgeDockerfile(PlatformTemplate):
    label = _(
        message='Template that generates a Mayan Forge Dockerfile file.'
    )
    name = 'forge_dockerfile'
    template_name = 'platform/forge/dockerfile.tmpl'

    def __init__(self):
        with open(file='docker/rootfs/version', mode='r') as file_object:
            version_string = file_object.readline()
            version_string_clean = version_string.strip()

        self.variables = (
            Variable(
                name='FORGE_DOCKER_BASE_IMAGE_NAME',
                default=FORGE_DOCKER_BASE_IMAGE_NAME,
                environment_name='MAYAN_FORGE_DOCKER_BASE_IMAGE_NAME'
            ),
            Variable(
                name='FORGE_DOCKER_BASE_IMAGE_TAG',
                default=FORGE_DOCKER_BASE_IMAGE_TAG,
                environment_name='MAYAN_FORGE_DOCKER_BASE_IMAGE_TAG'
            ),
            Variable(
                name='FORGE_PYTHON_PACKAGES',
                default=FORGE_PYTHON_PACKAGES,
                environment_name='MAYAN_FORGE_PYTHON_PACKAGES'
            ),
            Variable(
                name='FORGE_VIRTUALENV_SUFFIX',
                default=version_string_clean,
                environment_name='MAYAN_FORGE_VIRTUALENV_SUFFIX'
            ),
            Variable(
                name='FORGE_PYTHON_VERSION',
                default=FORGE_PYTHON_VERSION,
                environment_name='MAYAN_FORGE_PYTHON_VERSION'
            ),
            Variable(
                name='FORGE_TRANSIFEX_VERSION',
                default=FORGE_TRANSIFEX_VERSION,
                environment_name='MAYAN_FORGE_TRANSIFEX_VERSION'
            ),
            Variable(
                name='FORGE_VIRTUALENV_SUFFIX',
                default=version_string_clean,
                environment_name='MAYAN_FORGE_VIRTUALENV_SUFFIX'
            ),
            Variable(
                name='LINUX_PACKAGES_DEBIAN_BASE',
                default=LINUX_PACKAGES_DEBIAN_BASE,
                environment_name='MAYAN_FORGE_LINUX_PACKAGES_DEBIAN_BASE'
            ),
            Variable(
                name='LINUX_PACKAGES_DEBIAN_BUILD',
                default=LINUX_PACKAGES_DEBIAN_BUILD,
                environment_name='MAYAN_FORGE_LINUX_PACKAGES_DEBIAN_BUILD'
            ),
            Variable(
                name='LINUX_PACKAGES_DEBIAN_DOCUMENTATION',
                default=LINUX_PACKAGES_DEBIAN_DOCUMENTATION,
                environment_name='MAYAN_FORGE_LINUX_PACKAGES_DEBIAN_DOCUMENTATION'
            ),
            Variable(
                name='LINUX_PACKAGES_DEBIAN_FORGE',
                default=LINUX_PACKAGES_DEBIAN_FORGE,
                environment_name='MAYAN_LINUX_PACKAGES_DEBIAN_FORGE'
            ),
            Variable(
                name='LINUX_PACKAGES_DEBIAN_MYSQL',
                default=LINUX_PACKAGES_DEBIAN_MYSQL,
                environment_name='MAYAN_FORGE_LINUX_PACKAGES_DEBIAN_MYSQL'
            ),
            Variable(
                name='LINUX_PACKAGES_DEBIAN_POSTGRESQL',
                default=LINUX_PACKAGES_DEBIAN_POSTGRESQL,
                environment_name='MAYAN_FORGE_LINUX_PACKAGES_DEBIAN_POSTGRESQL'
            ),
            Variable(
                name='LINUX_PACKAGES_DEBIAN_PUSH',
                default=LINUX_PACKAGES_DEBIAN_PUSH,
                environment_name='MAYAN_FORGE_LINUX_PACKAGES_DEBIAN_PUSH'
            ),
            Variable(
                name='LINUX_PACKAGES_DEBIAN_PYTHON',
                default=LINUX_PACKAGES_DEBIAN_PYTHON,
                environment_name='MAYAN_FORGE_LINUX_PACKAGES_DEBIAN_PYTHON'
            ),
            Variable(
                name='LINUX_PACKAGES_DEBIAN_TEST',
                default=LINUX_PACKAGES_DEBIAN_TEST,
                environment_name='MAYAN_FORGE_LINUX_PACKAGES_DEBIAN_TEST'
            )
        )
