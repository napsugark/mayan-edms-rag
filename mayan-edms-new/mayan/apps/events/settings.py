from django.utils.translation import gettext_lazy as _

from mayan.apps.smart_settings.settings import setting_cluster

from .literals import (
    DEFAULT_EVENTS_DISABLE_ASYNCHRONOUS_MODE, DEFAULT_EVENTS_PRUNE_BACKEND,
    DEFAULT_EVENTS_PRUNE_BACKEND_ARGUMENTS,
    DEFAULT_EVENTS_PRUNE_TASK_INTERVAL
)

setting_namespace = setting_cluster.do_namespace_add(
    label=_(message='Events'), name='events'
)

setting_disable_asynchronous_mode = setting_namespace.do_setting_add(
    default=DEFAULT_EVENTS_DISABLE_ASYNCHRONOUS_MODE,
    global_name='EVENTS_DISABLE_ASYNCHRONOUS_MODE',
    help_text=_(
        message='Disables asynchronous events mode. All events will be '
        'committed in the same process that triggers them. This was the '
        'behavior prior to version 4.5.'
    )
)
setting_event_prune_backend = setting_namespace.do_setting_add(
    default=DEFAULT_EVENTS_PRUNE_BACKEND, global_name='EVENTS_PRUNE_BACKEND',
    help_text=_(
        'Path to the event pruning subclass that will be called '
        'periodically to clear the event log.'
    )
)
setting_event_prune_backend_arguments = setting_namespace.do_setting_add(
    default=DEFAULT_EVENTS_PRUNE_BACKEND_ARGUMENTS,
    global_name='EVENTS_PRUNE_BACKEND_ARGUMENTS',
    help_text=_('Arguments to pass to `EVENTS_PRUNE_BACKEND`.')
)
setting_event_prune_task_interval = setting_namespace.do_setting_add(
    default=DEFAULT_EVENTS_PRUNE_TASK_INTERVAL,
    global_name='EVENTS_PRUNE_TASK_INTERVAL',
    help_text=_(
        'Time interval in seconds, at which the event prune task will '
        'execute.'
    )
)
