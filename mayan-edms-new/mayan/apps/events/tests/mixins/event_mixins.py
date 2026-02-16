from django.db.models import F
from django.test import tag
from django.utils import timezone

from actstream.models import Action

from mayan.apps.acls.classes import ModelPermission
from mayan.apps.testing.tests.mixins import TestMixinObjectCreationTrack

from ...classes import EventModelRegistry, EventType, ModelEventType
from ...permissions import permission_events_view

from .event_type_mixins import EventTypeTestMixin


class EventsClearViewTestMixin:
    def _request_test_verb_event_list_clear_view(self):
        return self.post(
            viewname='events:verb_event_list_clear', kwargs={
                'verb': self._test_event_type.id
            }
        )

    def _request_test_event_list_clear_view(self):
        return self.post(viewname='events:event_list_clear')

    def _request_object_event_list_clear_view(self):
        return self.post(
            viewname='events:object_event_list_clear',
            kwargs=self.view_arguments
        )


class EventsExportViewTestMixin:
    def _request_test_verb_event_list_export_view(self):
        return self.post(
            viewname='events:verb_event_list_export', kwargs={
                'verb': self._test_event_type.id
            }
        )

    def _request_test_event_list_export_view(self):
        return self.post(viewname='events:event_list_export')

    def _request_object_event_list_export_view(self):
        return self.post(
            viewname='events:object_event_list_export',
            kwargs=self.view_arguments
        )


class EventListAPIViewTestMixin:
    def _request_test_event_list_api_view(self):
        return self.get(viewname='rest_api:event-list')


class EventTestCaseMixin:
    def setUp(self):
        # Simulate `EventTypeNamespace.post_load_modules` to ensure all
        # stored event types are properly initialized for each test.
        super().setUp()

        EventType.refresh()

        Action.objects.all().delete()

    def _clear_events(self):
        Action.objects.all().delete()

    def _get_test_events(self):
        return Action.objects.all().order_by('timestamp')


@tag('events')
class EventTestMixin(EventTypeTestMixin, TestMixinObjectCreationTrack):
    _test_object_model = Action
    _test_object_name = '_test_event'

    def setUp(self):
        super().setUp()
        self._test_event_list = []

    def _create_test_event(
        self, action_object=None, actor=None, event_type=None, target=None,
        timedelta=None
    ):
        self._test_object_track()

        event_type = event_type or self._test_event_type

        event_type.commit(
            action_object=action_object, actor=actor or self._test_case_user,
            target=target
        )

        self._test_object_set()

        if timedelta:
            queryset = Action.objects.filter(pk=self._test_event.pk)
            queryset.update(
                timestamp=F('timestamp') - timezone.timedelta(**timedelta)
            )

        self._test_event_list.append(self._test_event)


class EventObjectTestMixin(EventTestMixin):
    def _create_test_object_with_event_type_and_permission(self):
        self._create_test_object()

        EventModelRegistry.register(model=self._test_model_dict['_TestModel_0'])

        ModelEventType.register(
            event_types=(self._test_event_type,), model=self._test_model_dict['_TestModel_0']
        )

        ModelPermission.register(
            model=self._test_model_dict['_TestModel_0'], permissions=(
                permission_events_view,
            )
        )


class EventViewTestMixin(EventTestMixin):
    def _request_test_verb_event_list_view(self):
        return self.get(
            viewname='events:verb_event_list', kwargs={
                'verb': self._test_event_type.id
            }
        )

    def _request_test_event_list_view(self):
        return self.get(viewname='events:event_list')

    def _request_test_object_event_list_view(self):
        return self.get(
            viewname='events:object_event_list', kwargs=self.view_arguments
        )


class ObjectEventAPITestMixin:
    def _request_object_event_list_api_view(self):
        return self.get(
            viewname='rest_api:object-event-list',
            kwargs=self.view_arguments
        )


class UserEventViewTestMixin:
    def _request_test_user_event_type_subscription_list_view(self):
        return self.get(viewname='events:event_type_user_subscription_list')
