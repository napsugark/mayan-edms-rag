from django.utils.translation import gettext_lazy as _

DEFAULT_APT_INSTALLS = ''
DEFAULT_DOCKER_SCRIPT_POST_SETUP = ''
DEFAULT_DOCKER_SCRIPT_PRE_SETUP = ''
DEFAULT_DOCKER_WAIT = ''
DEFAULT_PIP_INSTALLS = ''
DEFAULT_SKIP_CHOWN_ON_STARTUP = False

worker_setting_map = (
    (
        'concurrency', {
            'default_name': 'CONCURRENCY',
            'help_text': _(
                message='Allows setting the worker\'s Celery `concurrency` '
                'value.'
            )
        }
    ),
    (
        'maximum_memory_per_child', {
            'default_name': 'MAX_MEMORY_PER_CHILD',
            'help_text': _(
                message='Allows setting the worker\'s Celery '
                '`max-memory-per-child` value.'
            )
        }
    ),
    (
        'maximum_tasks_per_child', {
            'default_name': 'MAX_TASKS_PER_CHILD',
            'help_text': _(
                message='Allows setting the worker\'s Celery '
                '`max-tasks-per-child` value.'
            )
        }
    ),
)
