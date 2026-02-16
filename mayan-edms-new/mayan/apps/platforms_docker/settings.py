from django.utils.translation import gettext_lazy as _

from mayan.apps.smart_settings.settings import setting_cluster
from mayan.apps.task_manager.classes import Worker
from mayan.settings.literals import (  # NOQA
    DOCKER_USER_GID, DOCKER_USER_UID, MAYAN_WORKER_A_CONCURRENCY,
    MAYAN_WORKER_A_MAX_MEMORY_PER_CHILD, MAYAN_WORKER_A_MAX_TASKS_PER_CHILD,
    MAYAN_WORKER_B_CONCURRENCY, MAYAN_WORKER_B_MAX_MEMORY_PER_CHILD,
    MAYAN_WORKER_B_MAX_TASKS_PER_CHILD, MAYAN_WORKER_C_CONCURRENCY,
    MAYAN_WORKER_C_MAX_MEMORY_PER_CHILD, MAYAN_WORKER_C_MAX_TASKS_PER_CHILD,
    MAYAN_WORKER_D_CONCURRENCY, MAYAN_WORKER_D_MAX_MEMORY_PER_CHILD,
    MAYAN_WORKER_D_MAX_TASKS_PER_CHILD, MAYAN_WORKER_E_CONCURRENCY,
    MAYAN_WORKER_E_MAX_MEMORY_PER_CHILD, MAYAN_WORKER_E_MAX_TASKS_PER_CHILD
)

from .literals import (
    DEFAULT_APT_INSTALLS, DEFAULT_DOCKER_SCRIPT_POST_SETUP,
    DEFAULT_DOCKER_SCRIPT_PRE_SETUP, DEFAULT_DOCKER_WAIT,
    DEFAULT_PIP_INSTALLS, DEFAULT_SKIP_CHOWN_ON_STARTUP,
    worker_setting_map
)

setting_namespace = setting_cluster.do_namespace_add(
    label=_(message='Docker'), name='docker'
)

setting_namespace.do_setting_add(
    default=DEFAULT_APT_INSTALLS, global_name='MAYAN_APT_INSTALLS',
    help_text=_(
        message='Specifies a list of .deb packages to be installed via '
        'APT when the container is first created. The installed packages '
        'are not lost when the image is stopped.'
    )
)
setting_pip_install = setting_namespace.do_setting_add(
    default=DEFAULT_PIP_INSTALLS, global_name='MAYAN_PIP_INSTALLS',
    help_text=_(
        message='Specifies a list of Python packages to be installed via '
        'pip. Packages will be downloaded from the Python Package Index '
        '(https://pypi.python.org) by default.'
    )
)
setting_script_post_setup = setting_namespace.do_setting_add(
    default=DEFAULT_DOCKER_SCRIPT_POST_SETUP,
    global_name='MAYAN_DOCKER_SCRIPT_POST_SETUP', help_text=_(
        message='Executed after the container\'s environment variables '
        'are configured, after the UID/GID setup, extra OS package '
        'installations and extra Python library installations but before '
        'launching the Mayan EDMS stack. Executes the content as a script '
        'or call a script using the value as the filename.'
    )
)
setting_script_pre_setup = setting_namespace.do_setting_add(
    default=DEFAULT_DOCKER_SCRIPT_PRE_SETUP,
    global_name='MAYAN_DOCKER_SCRIPT_PRE_SETUP', help_text=_(
        message='Executed after the container\'s environment variables are '
        'configured but before the UID/GID setup, extra OS package '
        'installations and extra Python library installations. Executes '
        'the content as a script or call a script using the value as the '
        'filename.'
    )
)
setting_namespace.do_setting_add(
    default=DEFAULT_SKIP_CHOWN_ON_STARTUP,
    global_name='MAYAN_SKIP_CHOWN_ON_STARTUP', help_text=_(
        message='Setting this environment variable to true, will make the '
        'entrypoint script skip the initial `chown` command on the media '
        'folder at `/var/lib/mayan`.'
    )
)
setting_namespace.do_setting_add(
    default=DOCKER_USER_GID,
    global_name='MAYAN_USER_GID', help_text=_(
        message='Changes the GID of the `mayan` user internal to the Docker '
        'container.'
    )
)
setting_namespace.do_setting_add(
    default=DOCKER_USER_UID,
    global_name='MAYAN_USER_UID', help_text=_(
        message='Changes the UID of the `mayan` user internal to the Docker '
        'container.'
    )
)

setting_namespace.do_setting_add(
    default=DEFAULT_DOCKER_WAIT,
    global_name='MAYAN_DOCKER_WAIT', help_text=_(
        message='Make the Docker container wait for a host and port to '
        'become available. Multiple hosts and port combinations are '
        'supported.'
    )
)

for worker in Worker.all():
    worker_name = worker.name.upper()

    for setting_name, settings_attributes in worker_setting_map:
        global_name = 'MAYAN_{}_{}'.format(
            worker_name, setting_name.upper()
        )

        default_name = 'MAYAN_{}_{}'.format(
            worker_name, settings_attributes['default_name'].upper()
        )

        default = globals()[default_name]

        help_text = settings_attributes['help_text']

        setting_namespace.do_setting_add(
            default=default, global_name=global_name, help_text=help_text
        )
