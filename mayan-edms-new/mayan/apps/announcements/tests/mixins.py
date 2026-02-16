from mayan.apps.testing.tests.mixins import TestMixinObjectCreationTrack

from ..models import Announcement

from .literals import (
    TEST_ANNOUNCEMENT_LABEL, TEST_ANNOUNCEMENT_LABEL_EDITED,
    TEST_ANNOUNCEMENT_TEXT, TEST_ANNOUNCEMENT_TEXT_EDITED
)


class AnnouncementTestMixin(TestMixinObjectCreationTrack):
    _test_object_model = Announcement
    _test_object_name = '_test_announcement'

    def _create_test_announcement(self):
        self._test_announcement = Announcement.objects.create(
            label=TEST_ANNOUNCEMENT_LABEL,
            text=TEST_ANNOUNCEMENT_TEXT
        )


class AnnouncementAPIViewTestMixin(AnnouncementTestMixin):
    def _request_announcement_create_view(self):
        self._test_object_track()

        response = self.post(
            viewname='rest_api:announcement-list', data={
                'label': TEST_ANNOUNCEMENT_LABEL,
                'text': TEST_ANNOUNCEMENT_TEXT
            }
        )

        self._test_object_set()

        return response

    def _request_announcement_delete_view(self):
        return self.delete(
            viewname='rest_api:announcement-detail', kwargs={
                'announcement_id': self._test_announcement.pk
            }
        )

    def _request_announcement_detail_view(self):
        return self.get(
            viewname='rest_api:announcement-detail', kwargs={
                'announcement_id': self._test_announcement.pk
            }
        )

    def _request_announcement_edit_via_patch_view(self):
        return self.patch(
            viewname='rest_api:announcement-detail', kwargs={
                'announcement_id': self._test_announcement.pk
            }, data={
                'label': TEST_ANNOUNCEMENT_LABEL_EDITED,
                'text': TEST_ANNOUNCEMENT_TEXT_EDITED
            }
        )

    def _request_announcement_edit_via_put_view(self):
        return self.put(
            viewname='rest_api:announcement-detail', kwargs={
                'announcement_id': self._test_announcement.pk
            }, data={
                'label': TEST_ANNOUNCEMENT_LABEL_EDITED,
                'text': TEST_ANNOUNCEMENT_TEXT_EDITED
            }
        )


class AnnouncementViewTestMixin(AnnouncementTestMixin):
    def _request_test_announcement_create_view(self):
        self._test_object_track()

        response = self.post(
            viewname='announcements:announcement_create', data={
                'label': TEST_ANNOUNCEMENT_LABEL,
                'text': TEST_ANNOUNCEMENT_TEXT
            }
        )

        self._test_object_set()

        return response

    def _request_test_announcement_delete_view(self):
        return self.post(
            viewname='announcements:announcement_single_delete', kwargs={
                'announcement_id': self._test_announcement.pk
            }
        )

    def _request_test_announcement_edit_view(self):
        return self.post(
            viewname='announcements:announcement_edit', kwargs={
                'announcement_id': self._test_announcement.pk
            }, data={
                'label': TEST_ANNOUNCEMENT_LABEL_EDITED,
                'text': TEST_ANNOUNCEMENT_TEXT_EDITED
            }
        )

    def _request_test_announcement_list_view(self):
        return self.get(viewname='announcements:announcement_list')
