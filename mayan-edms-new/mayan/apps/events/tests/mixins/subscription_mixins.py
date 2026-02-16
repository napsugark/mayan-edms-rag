from ...models import ObjectEventSubscription


class ObjectEventSubscriptionTestMixin:
    def _create_test_object_event_subscription(self):
        ObjectEventSubscription.objects.create(
            content_object=self._test_object,
            user=self._test_case_user,
            stored_event_type=self._test_event_type.get_stored_event_type()
        )


class UserObjectSubscriptionViewTestMixin:
    def _request_user_object_subscription_list_view(self):
        return self.get(
            viewname='events:user_object_subscription_list'
        )
