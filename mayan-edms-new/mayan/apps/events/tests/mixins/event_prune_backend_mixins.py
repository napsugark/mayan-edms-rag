from actstream.models import Action

from .event_mixins import EventTestMixin


class EventLogPruneBackendTestMixin(EventTestMixin):
    _test_event_prune_backend_class = None
    _test_event_prune_backend_kwargs = None

    def setUp(self):
        super().setUp()

        self._create_test_model()
        self._create_test_event_type(register=True)
        self._create_test_event_type(register=True)

        self._clear_events()

        self._create_test_object()

        self._create_test_event(
            event_type=self._test_event_type_list[0],
            target=self._test_object, timedelta={'days': 100}
        )
        self._create_test_event(
            event_type=self._test_event_type_list[0],
            target=self._test_object, timedelta={'days': 100}
        )
        self._create_test_event(
            event_type=self._test_event_type_list[1],
            target=self._test_object, timedelta={'days': 100}
        )
        self._create_test_event(
            event_type=self._test_event_type_list[1],
            target=self._test_object, timedelta={'days': 100}
        )
        self._create_test_event(
            event_type=self._test_event_type_list[1],
            target=self._test_object, timedelta={'days': 100}
        )
        self._create_test_event(
            event_type=self._test_event_type_list[1],
            target=self._test_object, timedelta={'days': 100}
        )

        self._create_test_object()

        self._create_test_event(
            event_type=self._test_event_type_list[1],
            target=self._test_object, timedelta={'days': 100}
        )
        self._create_test_event(
            event_type=self._test_event_type_list[1],
            target=self._test_object, timedelta={'days': 100}
        )
        self._create_test_event(
            event_type=self._test_event_type_list[0],
            target=self._test_object
        )
        self._create_test_event(
            event_type=self._test_event_type_list[0],
            target=self._test_object
        )
        self._create_test_event(
            event_type=self._test_event_type_list[0],
            target=self._test_object
        )
        self._create_test_event(
            event_type=self._test_event_type_list[0],
            target=self._test_object
        )

        self._test_event_count = Action.objects.count()

        self._test_object_0_event_count = self._test_object_list[0].target_actions.count()
        self._test_object_1_event_count = self._test_object_list[1].target_actions.count()

        kwargs = self._test_event_prune_backend_kwargs or {}

        _test_event_prune_backend = self._test_event_prune_backend_class(
            **kwargs
        )
        _test_event_prune_backend.execute()
