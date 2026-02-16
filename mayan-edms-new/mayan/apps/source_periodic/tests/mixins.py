from mayan.apps.sources.tests.literals import (
    TEST_CASE_ACTION_NAME_SOURCE_CREATE
)
from mayan.apps.sources.tests.mixins.base_mixins import SourceTestMixin

from ..source_backends.literals import DEFAULT_PERIOD_INTERVAL

from .literals import TEST_SOURCE_BACKEND_PATH_PERIODIC


class PeriodicSourceBackendTestMixin(SourceTestMixin):
    _test_source_backend_path = TEST_SOURCE_BACKEND_PATH_PERIODIC

    def _get_test_source_backend_data(self, action_name, interface_name):
        result = super()._get_test_source_backend_data(
            action_name=action_name, interface_name=interface_name
        )

        if action_name == TEST_CASE_ACTION_NAME_SOURCE_CREATE:
            result.update(
                {
                    'document_type_id': self._test_document_type.pk,
                    'interval': DEFAULT_PERIOD_INTERVAL
                }
            )

        return result
