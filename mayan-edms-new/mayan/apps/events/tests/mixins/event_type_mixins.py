from ...classes import EventModelRegistry, EventTypeNamespace, ModelEventType

from ..literals import (
    TEST_EVENT_TYPE_LABEL, TEST_EVENT_TYPE_NAME,
    TEST_EVENT_TYPE_NAMESPACE_LABEL, TEST_EVENT_TYPE_NAMESPACE_NAME
)


class EventTypeTestMixin:
    def setUp(self):
        super().setUp()
        self._test_event_type_list = []

        self._test_event_type_namespace = EventTypeNamespace(
            label=TEST_EVENT_TYPE_NAMESPACE_LABEL,
            name=TEST_EVENT_TYPE_NAMESPACE_NAME
        )

    def tearDown(self):
        for event_type in self._test_event_type_list.copy():
            ModelEventType.deregister_event_type(event_type=event_type)

            self._test_event_type_namespace.event_type_remove(
                event_type=event_type
            )
            self._test_event_type_list.remove(event_type)

        self._test_event_type_namespace.do_delete()

        super().tearDown()

    def _create_test_event_type(self, register=False, model=None):
        total_test_event_type_count = len(
            self._test_event_type_namespace.event_type_list
        )
        test_event_label = '{}_{}'.format(
            TEST_EVENT_TYPE_LABEL, total_test_event_type_count
        )
        test_event_name = '{}_{}'.format(
            TEST_EVENT_TYPE_NAME, total_test_event_type_count
        )

        self._test_event_type = self._test_event_type_namespace.add_event_type(
            label=test_event_label, name=test_event_name
        )

        # Initialize cache of the new event type.
        self._test_event_type.get_stored_event_type()

        self._test_event_type_list.append(self._test_event_type)

        if register:
            model = model or self._TestModel

            EventModelRegistry.register(model=model)

            ModelEventType.register(
                event_types=(self._test_event_type,), model=model
            )


class EventTypeNamespaceAPITestMixin(EventTypeTestMixin):
    def _request_test_event_type_list_api_view(self):
        return self.get(viewname='rest_api:event-type-list')

    def _request_test_event_namespace_list_api_view(self):
        return self.get(viewname='rest_api:event-type-namespace-list')

    def _request_test_event_type_namespace_event_type_list_api_view(self):
        return self.get(
            viewname='rest_api:event-type-namespace-event-type-list',
            kwargs={
                'name': self._test_event_type_namespace.name
            }
        )
