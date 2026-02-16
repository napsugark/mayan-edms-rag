from django.utils.translation import gettext_lazy as _

from mayan.apps.navigation.links import Link

from .icons import (
    icon_task_type_list, icon_worker_list, icon_worker_queue_list
)
from .permissions import permission_task_view

link_queue_task_type_list = Link(
    icon=icon_task_type_list, kwargs={
        'queue_name': 'resolved_object.name'
    }, permission=permission_task_view, text=_(message='Task type list'),
    view='task_manager:queue_task_type_list'
)
link_task_type_list = Link(
    icon=icon_task_type_list, permission=permission_task_view,
    text=_(message='Task type list'), view='task_manager:task_type_list'
)
link_worker_list = Link(
    icon=icon_worker_list, permission=permission_task_view,
    text=_(message='Worker list'), view='task_manager:worker_list'
)
link_worker_queue_list = Link(
    icon=icon_worker_queue_list, kwargs={
        'worker_name': 'resolved_object.name'
    }, permission=permission_task_view, text=_(message='Queue list'),
    view='task_manager:worker_queue_list'
)
