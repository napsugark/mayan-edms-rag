from django.utils.functional import cached_property

from .classes import EventType
from .literals import TEXT_UNKNOWN_EVENT_ID


class StoredEventTypeBusinessLogicMixin:
    @cached_property
    def event_type(self):
        return EventType.get(id=self.name)

    @property
    def label(self):
        try:
            event_type = self.event_type
        except KeyError:
            return TEXT_UNKNOWN_EVENT_ID % self.name
        else:
            return event_type.label

    @property
    def namespace(self):
        return self.event_type.namespace


class NotificationBusinessLogicMixin:
    def get_event_type(self):
        try:
            return EventType.get(id=self.action.verb)
        except KeyError:
            return None
