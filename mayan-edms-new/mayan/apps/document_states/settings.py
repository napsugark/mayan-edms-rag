from django.utils.translation import gettext_lazy as _

from mayan.apps.smart_settings.settings import setting_cluster

from .literals import (
    DEFAULT_GRAPHVIZ_DOT_PATH, DEFAULT_WORKFLOWS_IMAGE_CACHE_MAXIMUM_SIZE,
    DEFAULT_WORKFLOWS_IMAGE_CACHE_STORAGE_BACKEND,
    DEFAULT_WORKFLOWS_IMAGE_CACHE_STORAGE_BACKEND_ARGUMENTS,
    DEFAULT_WORKFLOWS_WORKFLOW_STATE_ESCALATION_CHECK_INTERVAL
)
from .setting_callbacks import callback_update_workflow_image_cache_size

setting_namespace = setting_cluster.do_namespace_add(
    label=_(message='Workflows'), name='document_states'
)

setting_graphviz_dot_path = setting_namespace.do_setting_add(
    default=DEFAULT_GRAPHVIZ_DOT_PATH,
    global_name='WORKFLOWS_GRAPHVIZ_DOT_PATH', help_text=_(
        message='File path to the graphviz dot program used to generate workflow '
        'previews.'
    ), is_path=True
)
setting_workflow_image_cache_maximum_size = setting_namespace.do_setting_add(
    default=DEFAULT_WORKFLOWS_IMAGE_CACHE_MAXIMUM_SIZE,
    global_name='WORKFLOWS_IMAGE_CACHE_MAXIMUM_SIZE',
    help_text=_(
        message='The threshold at which the WORKFLOWS_IMAGE_CACHE_STORAGE_BACKEND '
        'will start deleting the oldest workflow image cache files. '
        'Specify the size in bytes.'
    ), post_edit_function=callback_update_workflow_image_cache_size
)

setting_workflow_image_cache_storage_backend = setting_namespace.do_setting_add(
    default=DEFAULT_WORKFLOWS_IMAGE_CACHE_STORAGE_BACKEND,
    global_name='WORKFLOWS_IMAGE_CACHE_STORAGE_BACKEND', help_text=_(
        message='Path to the Storage subclass to use when storing the cached '
        'workflow image files.'
    )
)
setting_workflow_image_cache_storage_backend_arguments = setting_namespace.do_setting_add(
    default=DEFAULT_WORKFLOWS_IMAGE_CACHE_STORAGE_BACKEND_ARGUMENTS,
    global_name='WORKFLOWS_IMAGE_CACHE_STORAGE_BACKEND_ARGUMENTS',
    help_text=_(
        message='Arguments to pass to the WORKFLOWS_IMAGE_CACHE_STORAGE_BACKEND.'
    )
)
setting_workflow_state_escalation_check_interval = setting_namespace.do_setting_add(
    default=DEFAULT_WORKFLOWS_WORKFLOW_STATE_ESCALATION_CHECK_INTERVAL,
    global_name='WORKFLOWS_WORKFLOW_STATE_ESCALATION_CHECK_INTERVAL',
    help_text=_(
        message='Interval in seconds on which the task to check for expired '
        'workflow states will be launched.'
    )
)
