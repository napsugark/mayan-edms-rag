from mayan.apps.credentials.tests.mixins import (
    StoredCredentialPasswordUsernameTestMixin
)
from mayan.apps.testing.tests.base import BaseTestCase

from .mixins import SourceTestMixinStoredFile


class SourceBackendStorageTestCase(SourceTestMixinStoredFile, BaseTestCase):
    _test_source_create_auto = False

    def test_storage_arguments_template_default(self):
        TEST_SOURCE_STORAGE_BACKEND_ARGUMENTS_TEXT = '{\'location\': \'\'}'
        TEST_SOURCE_STORAGE_BACKEND_ARGUMENTS_OBJECT = {'location': ''}

        self._test_source_create(
            extra_data={
                'storage_backend_arguments': TEST_SOURCE_STORAGE_BACKEND_ARGUMENTS_TEXT
            }
        )

        backend_instance = self._test_source.get_backend_instance()

        storage_backend_arguments = backend_instance.get_storage_backend_arguments()

        self.assertEqual(
            storage_backend_arguments,
            TEST_SOURCE_STORAGE_BACKEND_ARGUMENTS_OBJECT
        )


class SourceBackendStorageCredentialTestCase(
    SourceTestMixinStoredFile, StoredCredentialPasswordUsernameTestMixin,
    BaseTestCase
):
    _test_source_create_auto = False

    def test_storage_arguments_credential_template(self):
        TEST_SOURCE_STORAGE_BACKEND_ARGUMENTS_TEXT = '{\'username\': \'{{ credential.username }}\', \'password\': \'{{ credential.password }}\'}'
        TEST_SOURCE_STORAGE_BACKEND_ARGUMENTS_OBJECT = {
            'password': self._test_stored_credential._password,
            'username': self._test_stored_credential._username
        }

        self._test_source_create(
            extra_data={
                'stored_credential_id': self._test_stored_credential.pk,
                'storage_backend_arguments': TEST_SOURCE_STORAGE_BACKEND_ARGUMENTS_TEXT
            }
        )

        backend_instance = self._test_source.get_backend_instance()
        storage_backend_arguments = backend_instance.get_storage_backend_arguments()

        self.assertEqual(
            storage_backend_arguments,
            TEST_SOURCE_STORAGE_BACKEND_ARGUMENTS_OBJECT
        )
