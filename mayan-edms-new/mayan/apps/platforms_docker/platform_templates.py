from django.utils.translation import gettext_lazy as _

from mayan.apps.platforms.platform_templates import (
    PlatformTemplate, PlatformTemplateSupervisord, Variable
)
from mayan.apps.platforms.utils import load_env_file
from mayan.apps.task_manager.classes import Worker
from mayan.settings.literals import (
    DEFAULT_DATABASE_NAME, DEFAULT_DATABASE_PASSWORD, DEFAULT_DATABASE_USER,
    DEFAULT_ELASTICSEARCH_PASSWORD, DEFAULT_RABBITMQ_CONSUMER_TIMEOUT,
    DEFAULT_RABBITMQ_PASSWORD, DEFAULT_RABBITMQ_USER, DEFAULT_RABBITMQ_VHOST,
    DEFAULT_REDIS_PASSWORD, DOCKER_ELASTIC_IMAGE_NAME,
    DOCKER_ELASTIC_IMAGE_TAG, DOCKER_IMAGE_MAYAN_NAME, DOCKER_IMAGE_MAYAN_TAG,
    DOCKER_LINUX_IMAGE_NAME, DOCKER_LINUX_IMAGE_TAG,
    DOCKER_POSTGRESQL_IMAGE_NAME, DOCKER_POSTGRESQL_IMAGE_TAG,
    DOCKER_POSTGRESQL_MAX_CONNECTIONS, DOCKER_RABBITMQ_IMAGE_NAME,
    DOCKER_RABBITMQ_IMAGE_TAG, DOCKER_REDIS_IMAGE_NAME,
    DOCKER_REDIS_IMAGE_TAG, DOCKER_TRAEFIK_IMAGE_NAME,
    DOCKER_TRAEFIK_IMAGE_TAG, LINUX_PACKAGES_DEBIAN_BASE,
    LINUX_PACKAGES_DEBIAN_BUILD, LINUX_PACKAGES_DEBIAN_MYSQL,
    LINUX_PACKAGES_DEBIAN_POSTGRESQL, LINUX_PACKAGES_DEBIAN_PYTHON
)


class PlatformTemplateDockerEntrypoint(PlatformTemplate):
    label = _(message='Template for entrypoint.sh file inside a Docker image.')
    name = 'docker_entrypoint'
    template_name = 'platforms_docker/entrypoint.tmpl'

    def get_context(self):
        context = load_env_file()
        context.update(
            {
                'SUPERVISOR_AUTORESTART': 'false',
                'workers': Worker.all()
            }
        )
        return context


class PlatformTemplateDockerComposefile(PlatformTemplate):
    label = _(message='Template that generates the Docker Compose file.')
    name = 'docker_docker_compose'
    template_name = 'platforms_docker/docker-compose.yml.tmpl'

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
                name='DEFAULT_ELASTICSEARCH_PASSWORD',
                default=DEFAULT_ELASTICSEARCH_PASSWORD,
                environment_name='MAYAN_DEFAULT_ELASTICSEARCH_PASSWORD'
            ),
            Variable(
                name='DEFAULT_RABBITMQ_CONSUMER_TIMEOUT',
                default=DEFAULT_RABBITMQ_CONSUMER_TIMEOUT,
                environment_name='MAYAN_DEFAULT_RABBITMQ_CONSUMER_TIMEOUT'
            ),
            Variable(
                name='DEFAULT_RABBITMQ_PASSWORD',
                default=DEFAULT_RABBITMQ_PASSWORD,
                environment_name='MAYAN_DEFAULT_RABBITMQ_PASSWORD'
            ),
            Variable(
                name='DEFAULT_RABBITMQ_USER',
                default=DEFAULT_RABBITMQ_USER,
                environment_name='MAYAN_DEFAULT_RABBITMQ_USER'
            ),
            Variable(
                name='DEFAULT_RABBITMQ_VHOST',
                default=DEFAULT_RABBITMQ_VHOST,
                environment_name='MAYAN_DEFAULT_RABBITMQ_VHOST'
            ),
            Variable(
                name='DEFAULT_REDIS_PASSWORD',
                default=DEFAULT_REDIS_PASSWORD,
                environment_name='MAYAN_DEFAULT_REDIS_PASSWORD'
            ),
            Variable(
                name='DOCKER_ELASTIC_IMAGE_NAME',
                default=DOCKER_ELASTIC_IMAGE_NAME,
                environment_name='MAYAN_DOCKER_ELASTIC_IMAGE_NAME'
            ),
            Variable(
                name='DOCKER_ELASTIC_IMAGE_TAG',
                default=DOCKER_ELASTIC_IMAGE_TAG,
                environment_name='MAYAN_DOCKER_ELASTIC_IMAGE_TAG'
            ),
            Variable(
                name='DOCKER_IMAGE_MAYAN_NAME',
                default=DOCKER_IMAGE_MAYAN_NAME,
                environment_name='MAYAN_DOCKER_IMAGE_MAYAN_NAME'
            ),
            Variable(
                name='DOCKER_IMAGE_MAYAN_TAG',
                default=DOCKER_IMAGE_MAYAN_TAG,
                environment_name='MAYAN_DOCKER_IMAGE_MAYAN_TAG'
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
                name='DOCKER_POSTGRESQL_MAX_CONNECTIONS',
                default=DOCKER_POSTGRESQL_MAX_CONNECTIONS,
                environment_name='MAYAN_DOCKER_POSTGRESQL_MAX_CONNECTIONS'
            ),
            Variable(
                name='DOCKER_RABBITMQ_IMAGE_NAME',
                default=DOCKER_RABBITMQ_IMAGE_NAME,
                environment_name='MAYAN_DOCKER_RABBITMQ_IMAGE_NAME'
            ),
            Variable(
                name='DOCKER_RABBITMQ_IMAGE_TAG',
                default=DOCKER_RABBITMQ_IMAGE_TAG,
                environment_name='MAYAN_DOCKER_RABBITMQ_IMAGE_TAG'
            ),
            Variable(
                name='DOCKER_REDIS_IMAGE_NAME',
                default=DOCKER_REDIS_IMAGE_NAME,
                environment_name='MAYAN_DOCKER_REDIS_IMAGE_NAME'
            ),
            Variable(
                name='DOCKER_REDIS_IMAGE_TAG',
                default=DOCKER_REDIS_IMAGE_TAG,
                environment_name='MAYAN_DOCKER_REDIS_IMAGE_TAG'
            ),
            Variable(
                name='DOCKER_TRAEFIK_IMAGE_NAME',
                default=DOCKER_TRAEFIK_IMAGE_NAME,
                environment_name='MAYAN_DOCKER_TRAEFIK_IMAGE_NAME'
            ),
            Variable(
                name='DOCKER_TRAEFIK_IMAGE_TAG',
                default=DOCKER_TRAEFIK_IMAGE_TAG,
                environment_name='MAYAN_DOCKER_TRAEFIK_IMAGE_TAG'
            ),
        )


class PlatformTemplateDockerSupervisord(PlatformTemplateSupervisord):
    label = _(message='Template for Supervisord inside a Docker image.')
    name = 'docker_supervisord'
    template_name = 'platforms_docker/supervisord.tmpl'

    def get_context(self):
        return {
            'stderr_logfile': '/dev/fd/2',
            'stderr_logfile_maxbytes': '0',
            'stdout_logfile': '/dev/fd/1',
            'stdout_logfile_maxbytes': '0',
            'workers': Worker.all()
        }


class PlatformTemplateDockerfile(PlatformTemplate):
    label = _(message='Template that generates a Dockerfile file.')
    name = 'docker_dockerfile'
    template_name = 'platforms_docker/dockerfile.tmpl'

    def __init__(self):
        self.variables = (
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
        )

    def get_context(self):
        return {
            'LINUX_PACKAGES_DEBIAN_BASE': LINUX_PACKAGES_DEBIAN_BASE,
            'LINUX_PACKAGES_DEBIAN_BUILD': LINUX_PACKAGES_DEBIAN_BUILD,
            'LINUX_PACKAGES_DEBIAN_MYSQL': LINUX_PACKAGES_DEBIAN_MYSQL,
            'LINUX_PACKAGES_DEBIAN_POSTGRESQL': LINUX_PACKAGES_DEBIAN_POSTGRESQL,
            'LINUX_PACKAGES_DEBIAN_PYTHON': LINUX_PACKAGES_DEBIAN_PYTHON
        }
