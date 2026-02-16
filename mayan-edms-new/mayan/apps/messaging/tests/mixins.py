from mayan.apps.testing.tests.mixins import TestMixinObjectCreationTrack

from ..models import Message

from .literals import TEST_MESSAGE_BODY, TEST_MESSAGE_SUBJECT


class MessageTestMixin(TestMixinObjectCreationTrack):
    _test_object_model = Message
    _test_object_name = '_test_message'
    auto_create_test_message = True

    def setUp(self):
        super().setUp()

        if self.auto_create_test_message:
            self._create_test_message()

    def _create_test_message(self):
        self._test_message = Message.objects.create(
            body=TEST_MESSAGE_BODY, subject=TEST_MESSAGE_SUBJECT,
            user=self._test_case_user
        )


class MessageAPIViewTestMixin(MessageTestMixin):
    def _request_test_message_create_api_view(self, extra_data=None):
        self._test_object_track()

        data = {
            'body': TEST_MESSAGE_BODY,
            'subject': TEST_MESSAGE_SUBJECT,
            'user': self._test_case_user.pk
        }

        if extra_data:
            data.update(extra_data)

        response = self.post(
            viewname='rest_api:message-list', data=data
        )

        self._test_object_set()

        return response

    def _request_test_message_delete_api_view(self):
        return self.delete(
            viewname='rest_api:message-detail',
            kwargs={'message_id': self._test_message.pk}
        )

    def _request_test_message_detail_api_view(self):
        return self.get(
            viewname='rest_api:message-detail',
            kwargs={'message_id': self._test_message.pk}
        )

    def _request_test_message_edit_api_view(
        self, extra_data=None, verb='patch'
    ):
        data = {
            'read': True,
        }

        if extra_data:
            data.update(extra_data)

        return getattr(self, verb)(
            viewname='rest_api:message-detail',
            kwargs={'message_id': self._test_message.pk},
            data=data
        )

    def _request_test_message_list_api_view(self):
        return self.get(viewname='rest_api:message-list')


class MessageViewTestMixin(MessageTestMixin):
    def _request_test_message_create_view(self, extra_data=None):
        self._test_object_track()

        data = {
            'body': TEST_MESSAGE_BODY,
            'subject': TEST_MESSAGE_SUBJECT,
            'user': self._test_case_user.pk
        }

        if extra_data:
            data.update(extra_data)

        response = self.post(
            viewname='messaging:message_create', data=data
        )

        self._test_object_set()

        return response

    def _request_test_message_delete_view(self):
        return self.post(
            viewname='messaging:message_single_delete', kwargs={
                'message_id': self._test_message.pk
            }
        )

    def _request_test_message_detail_view(self):
        return self.get(
            viewname='messaging:message_detail', kwargs={
                'message_id': self._test_message.pk
            }
        )

    def _request_test_message_list_view(self):
        return self.get(viewname='messaging:message_list')

    def _request_test_message_mark_all_read_view(self):
        return self.post(viewname='messaging:message_all_mark_read')

    def _request_test_message_mark_read_view(self):
        return self.post(
            viewname='messaging:message_single_mark_read', kwargs={
                'message_id': self._test_message.pk
            }
        )

    def _request_test_message_mark_unread_view(self):
        return self.post(
            viewname='messaging:message_single_mark_unread', kwargs={
                'message_id': self._test_message.pk
            }
        )
