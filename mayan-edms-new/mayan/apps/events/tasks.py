from django.apps import apps
from django.contrib.auth import get_user_model
from django.db import OperationalError

from mayan.apps.databases.classes import QuerysetParametersSerializer
from mayan.celery import app

from .classes import ActionExporter, EventType
from .event_prune_backends import EventLogPruneBackend
from .events import event_events_cleared
from .permissions import permission_events_clear
from .settings import setting_event_prune_backend


@app.task(bind=True, ignore_result=True, retry_backoff=True)
def task_event_commit(
    self, event_id, actor_app_label=None, actor_model_name=None,
    actor_id=None, action_object_app_label=None,
    action_object_model_name=None, action_object_id=None,
    target_app_label=None, target_model_name=None, target_id=None
):
    event_type = EventType.get(id=event_id)

    try:
        if action_object_id:
            Model = apps.get_model(
                app_label=action_object_app_label,
                model_name=action_object_model_name
            )

            action_object = Model.objects.get(pk=action_object_id)
        else:
            action_object = None

        if actor_id:
            Model = apps.get_model(
                app_label=actor_app_label, model_name=actor_model_name
            )

            actor = Model.objects.get(pk=actor_id)
        else:
            actor = None

        if target_id:
            Model = apps.get_model(
                app_label=target_app_label,
                model_name=target_model_name
            )

            target = Model.objects.get(pk=target_id)
        else:
            target = None

        event_type._commit(
            action_object=action_object, actor=actor, target=target
        )
    except OperationalError as exception:
        raise self.retry(exc=exception)


@app.task(ignore_result=True)
def task_event_prune():
    if setting_event_prune_backend.value:
        backend_instance = EventLogPruneBackend.get_backend_instance()
        backend_instance.execute()


@app.task(ignore_result=True)
def task_event_queryset_clear(
    decomposed_queryset, target_content_type_id=None, target_object_id=None,
    user_id=None
):
    AccessControlList = apps.get_model(
        app_label='acls', model_name='AccessControlList'
    )
    ContentType = apps.get_model(
        app_label='contenttypes', model_name='ContentType'
    )

    queryset = QuerysetParametersSerializer.rebuild(
        decomposed_queryset=decomposed_queryset
    )

    if user_id:
        user = get_user_model().objects.get(pk=user_id)
    else:
        user = None

    if target_content_type_id:
        target_content_type = ContentType.objects.get(
            pk=target_content_type_id
        )
        target = target_content_type.get_object_for_this_type(
            pk=target_object_id
        )
    else:
        target = user

    if user:
        queryset = AccessControlList.objects.restrict_queryset(
            queryset=queryset, permission=permission_events_clear,
            user=user
        )

    commit_event = queryset.exists()

    queryset.delete()

    if commit_event:
        event_events_cleared.commit(actor=user, target=target)


@app.task(ignore_result=True)
def task_event_queryset_export(
    decomposed_queryset, organization_installation_url=None, user_id=None
):
    queryset = QuerysetParametersSerializer.rebuild(
        decomposed_queryset=decomposed_queryset
    )

    if user_id:
        user = get_user_model().objects.get(pk=user_id)
    else:
        user = None

    ActionExporter(queryset=queryset).export_to_download_file(
        organization_installation_url=organization_installation_url,
        user=user
    )
