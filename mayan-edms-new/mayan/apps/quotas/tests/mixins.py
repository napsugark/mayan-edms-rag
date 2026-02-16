import json

from mayan.apps.documents.tests.base import DocumentTestMixin
from mayan.apps.source_web_forms.tests.mixins import WebFormSourceTestMixin
from mayan.apps.sources.tests.mixins.source_view_mixins import (
    SourceActionViewTestMixin
)
from mayan.apps.testing.tests.mixins import TestMixinObjectCreationTrack

from ..models import Quota

from .literals import (
    TEST_QUOTA_DATA, TEST_QUOTA_DOTTED_PATH, TEST_QUOTA_TEST_LIMIT_EDITED,
    TEST_QUOTA_WITH_MIXINS_DOTTED_PATH
)


class QuotaTestMixin(TestMixinObjectCreationTrack):
    _test_object_model = Quota
    _test_object_name = '_test_quota'

    def _create_test_quota(self):
        self._test_quota = Quota.objects.create(
            backend_data=json.dumps(obj=TEST_QUOTA_DATA),
            backend_path=TEST_QUOTA_DOTTED_PATH
        )

    def _create_test_quota_with_mixins(self):
        self._test_quota = Quota.objects.create(
            backend_data=json.dumps(obj=TEST_QUOTA_DATA),
            backend_path=TEST_QUOTA_WITH_MIXINS_DOTTED_PATH
        )


class DocumentViewQuotaHookTestMixin(
    DocumentTestMixin, SourceActionViewTestMixin,
    WebFormSourceTestMixin
):
    """
    Combined class for source document upload view quota testing.
    """


class QuotaViewTestMixin(QuotaTestMixin):
    def _request_test_quota_backend_selection_get_view(self):
        return self.get(viewname='quotas:quota_backend_selection')

    def _request_test_quota_create_get_view(self):
        self._test_object_track()

        response = self.get(
            viewname='quotas:quota_create', kwargs={
                'class_path': TEST_QUOTA_DOTTED_PATH
            }
        )

        self._test_object_set()

        return response

    def _request_test_quota_with_mixins_create_get_view(self):
        self._test_object_track()

        response = self.get(
            viewname='quotas:quota_create', kwargs={
                'class_path': TEST_QUOTA_WITH_MIXINS_DOTTED_PATH
            }
        )

        self._test_object_set()

        return response

    def _request_test_quota_create_post_view(self):
        self._test_object_track()

        response = self.post(
            viewname='quotas:quota_create', kwargs={
                'class_path': TEST_QUOTA_DOTTED_PATH
            }, data=TEST_QUOTA_DATA
        )

        self._test_object_set()

        return response

    def _request_test_quota_delete_view(self):
        return self.post(
            viewname='quotas:quota_delete', kwargs={
                'quota_id': self._test_quota.pk
            }
        )

    def _request_test_quota_edit_view(self):
        return self.post(
            viewname='quotas:quota_edit', kwargs={
                'quota_id': self._test_quota.pk
            }, data={
                'test_limit': TEST_QUOTA_TEST_LIMIT_EDITED
            }
        )

    def _request_test_quota_list_view(self):
        return self.get(viewname='quotas:quota_list')
