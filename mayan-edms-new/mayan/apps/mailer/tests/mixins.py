import json

from django.core.files.base import ContentFile

from mayan.apps.credentials.tests.mixins import (
    StoredCredentialPasswordUsernameTestMixin
)
from mayan.apps.events.classes import EventModelRegistry
from mayan.apps.testing.tests.mixins import TestMixinObjectCreationTrack

from ..mailing_actions import (
    ModelMailingActionAttachment, ModelMailingActionLink
)
from ..models import UserMailer

from .literals import (
    TEST_EMAIL_ADDRESS, TEST_EMAIL_FROM_ADDRESS, TEST_MAILING_OBJECT_CONTENT,
    TEST_MAILING_OBJECT_MIME_TYPE, TEST_MAILING_PROFILE_BACKEND_PATH,
    TEST_MAILING_PROFILE_LABEL, TEST_MAILING_PROFILE_LABEL_EDITED
)


class MailingProfileTestMixin(
    StoredCredentialPasswordUsernameTestMixin, TestMixinObjectCreationTrack
):
    _test_mailing_profile_auto_create = False
    _test_mailing_profile_backend_path = TEST_MAILING_PROFILE_BACKEND_PATH
    _test_object_model = UserMailer
    _test_object_name = '_test_mailing_profile'

    def setUp(self):
        super().setUp()

        if self._test_mailing_profile_auto_create:
            self._create_test_mailing_profile()

    def _create_test_mailing_profile(self, extra_backend_data=None):
        backend_data = {'from': TEST_EMAIL_FROM_ADDRESS}

        if extra_backend_data:
            backend_data.update(**extra_backend_data)

        self._test_mailing_profile = UserMailer.objects.create(
            default=True, enabled=True, label=TEST_MAILING_PROFILE_LABEL,
            backend_path=self._test_mailing_profile_backend_path,
            backend_data=json.dumps(obj=backend_data)
        )


class DocumentMailingProfileViewTestMixin(MailingProfileTestMixin):
    def _request_test_document_link_send_single_view(self):
        return self.post(
            viewname='mailer:send_document_link_single', kwargs={
                'document_id': self._test_document.pk
            }, data={
                'email': getattr(
                    self, '_test_email_address', TEST_EMAIL_ADDRESS
                ),
                'mailing_profile': self._test_mailing_profile.pk
            },
        )

    def _request_test_document_link_send_multiple_view(self):
        return self.post(
            viewname='mailer:send_document_link_multiple', query={
                'id_list': self._test_document.pk
            }, data={
                'email': getattr(
                    self, '_test_email_address', TEST_EMAIL_ADDRESS
                ),
                'mailing_profile': self._test_mailing_profile.pk
            }
        )


class DocumentFileMailingProfileViewTestMixin(MailingProfileTestMixin):
    def _request_test_document_file_link_send_single_view(self):
        return self.post(
            viewname='mailer:send_document_file_link_single', kwargs={
                'document_file_id': self._test_document_file.pk
            }, data={
                'email': getattr(
                    self, '_test_email_address', TEST_EMAIL_ADDRESS
                ),
                'mailing_profile': self._test_mailing_profile.pk
            }
        )

    def _request_test_document_file_link_send_multiple_view(self):
        return self.post(
            viewname='mailer:send_document_file_link_multiple', query={
                'id_list': self._test_document_file.pk
            }, data={
                'email': getattr(
                    self, '_test_email_address', TEST_EMAIL_ADDRESS
                ),
                'mailing_profile': self._test_mailing_profile.pk
            }
        )

    def _request_test_document_file_attachment_send_single_view(self):
        return self.post(
            viewname='mailer:send_document_file_attachment_single', kwargs={
                'document_file_id': self._test_document_file.pk
            }, data={
                'email': getattr(
                    self, '_test_email_address', TEST_EMAIL_ADDRESS
                ),
                'mailing_profile': self._test_mailing_profile.pk
            }
        )

    def _request_test_document_file_attachment_send_multiple_view(self):
        return self.post(
            viewname='mailer:send_document_file_attachment_multiple', query={
                'id_list': self._test_document_file.pk
            }, data={
                'email': getattr(
                    self, '_test_email_address', TEST_EMAIL_ADDRESS
                ),
                'mailing_profile': self._test_mailing_profile.pk
            }
        )


class DocumentVersionMailingProfileViewTestMixin(MailingProfileTestMixin):
    def _request_test_document_version_link_send_single_view(self):
        return self.post(
            viewname='mailer:send_document_version_link_single', kwargs={
                'document_version_id': self._test_document_version.pk
            }, data={
                'email': getattr(
                    self, '_test_email_address', TEST_EMAIL_ADDRESS
                ),
                'mailing_profile': self._test_mailing_profile.pk
            }
        )

    def _request_test_document_version_link_send_multiple_view(self):
        return self.post(
            viewname='mailer:send_document_version_link_multiple', query={
                'id_list': self._test_document_version.pk
            }, data={
                'email': getattr(
                    self, '_test_email_address', TEST_EMAIL_ADDRESS
                ),
                'mailing_profile': self._test_mailing_profile.pk
            }
        )

    def _request_test_document_version_attachment_send_single_view(self):
        return self.post(
            viewname='mailer:send_document_version_attachment_single',
            kwargs={
                'document_version_id': self._test_document_version.pk
            }, data={
                'email': getattr(
                    self, '_test_email_address', TEST_EMAIL_ADDRESS
                ),
                'mailing_profile': self._test_mailing_profile.pk
            }
        )

    def _request_test_document_version_attachment_send_multiple_view(self):
        return self.post(
            viewname='mailer:send_document_version_attachment_multiple',
            query={
                'id_list': self._test_document_version.pk
            }, data={
                'email': getattr(
                    self, '_test_email_address', TEST_EMAIL_ADDRESS
                ),
                'mailing_profile': self._test_mailing_profile.pk
            }
        )


def _test_content_function(obj):
    return ContentFile(content=TEST_MAILING_OBJECT_CONTENT)


def _test_mime_type_function(obj):
    return TEST_MAILING_OBJECT_MIME_TYPE


class MailObjectSendAPIViewTestMixin(MailingProfileTestMixin):
    _test_model_mailing_content_function_dotted_path = 'mayan.apps.mailer.tests.mixins._test_content_function'
    _test_model_mailing_mime_type_function_dotted_path = 'mayan.apps.mailer.tests.mixins._test_mime_type_function'
    _test_model_mailing_permission_attachment = None
    _test_model_mailing_permission_link = None

    def setUp(self):
        super().setUp()

        if self.auto_create_test_object:
            EventModelRegistry.register(
                model=self._test_model_dict['_TestModel_0']
            )

            self._test_model_dict['_TestModel_0'].add_to_class(
                name='get_absolute_url', value=lambda self: None
            )

            if self._test_model_mailing_permission_attachment:
                ModelMailingActionAttachment(
                    content_function_dotted_path=self._test_model_mailing_content_function_dotted_path,
                    manager_name='objects',
                    mime_type_function_dotted_path=self._test_model_mailing_mime_type_function_dotted_path,
                    model=self._test_model_dict['_TestModel_0'],
                    permission=self._test_model_mailing_permission_attachment
                )

            if self._test_model_mailing_permission_link:
                ModelMailingActionLink(
                    manager_name='objects',
                    model=self._test_model_dict['_TestModel_0'],
                    permission=self._test_model_mailing_permission_link
                )

    def _request_object_mail_attachment_api_view(self):
        return self.post(
            viewname='rest_api:object-mailing-action-attachment',
            kwargs=self._test_object_view_kwargs,
            data={
                'mailing_profile': self._test_mailing_profile.pk,
                'recipient': TEST_EMAIL_ADDRESS
            }
        )

    def _request_object_mail_link_api_view(self):
        return self.post(
            viewname='rest_api:object-mailing-action-link',
            kwargs=self._test_object_view_kwargs,
            data={
                'mailing_profile': self._test_mailing_profile.pk,
                'recipient': TEST_EMAIL_ADDRESS
            }
        )


class MailingProfileAPIViewTestMixin(MailingProfileTestMixin):
    def _request_test_mailing_profile_create_api_view(self):
        self._test_object_track()

        response = self.post(
            viewname='rest_api:mailing_profile-list', data={
                'backend_path': TEST_MAILING_PROFILE_BACKEND_PATH,
                'label': TEST_MAILING_PROFILE_LABEL
            }
        )

        self._test_object_set()

        return response

    def _request_test_mailing_profile_delete_api_view(self):
        return self.delete(
            viewname='rest_api:mailing_profile-detail',
            kwargs={'mailing_profile_id': self._test_mailing_profile.pk}
        )

    def _request_test_mailing_profile_detail_api_view(self):
        return self.get(
            viewname='rest_api:mailing_profile-detail',
            kwargs={'mailing_profile_id': self._test_mailing_profile.pk}
        )

    def _request_test_mailing_profile_edit_api_view(
        self, extra_data=None, verb='patch'
    ):
        data = {
            'backend_path': TEST_MAILING_PROFILE_BACKEND_PATH,
            'label': TEST_MAILING_PROFILE_LABEL_EDITED
        }

        if extra_data:
            data.update(extra_data)

        return getattr(self, verb)(
            viewname='rest_api:mailing_profile-detail',
            kwargs={'mailing_profile_id': self._test_mailing_profile.pk},
            data=data
        )

    def _request_test_mailing_profile_list_api_view(self):
        return self.get(viewname='rest_api:mailing_profile-list')


class MailingProfileViewTestMixin(MailingProfileTestMixin):
    def _request_test_mailing_profile_create_view(self):
        self._test_object_track()

        response = self.post(
            viewname='mailer:mailing_profile_create', kwargs={
                'backend_path': TEST_MAILING_PROFILE_BACKEND_PATH
            }, data={
                'default': True,
                'enabled': True,
                'label': TEST_MAILING_PROFILE_LABEL
            }
        )

        self._test_object_set()

        return response

    def _request_test_mailing_profile_delete_view(self):
        return self.post(
            viewname='mailer:mailing_profile_delete', kwargs={
                'mailing_profile_id': self._test_mailing_profile.pk
            }
        )

    def _request_test_mailing_profile_edit_view(self):
        return self.post(
            viewname='mailer:mailing_profile_edit', kwargs={
                'mailing_profile_id': self._test_mailing_profile.pk
            }, data={
                'label': '{}_edited'.format(TEST_MAILING_PROFILE_LABEL)
            }
        )

    def _request_test_mailing_profile_list_view(self):
        return self.get(
            viewname='mailer:mailing_profile_list'
        )

    def _request_test_mailing_profile_log_entry_view(self):
        return self.get(
            viewname='mailer:mailing_profile_log', kwargs={
                'mailing_profile_id': self._test_mailing_profile.pk
            }
        )

    def _request_test_mailing_profile_test_view(self):
        return self.post(
            viewname='mailer:mailing_profile_test', kwargs={
                'mailing_profile_id': self._test_mailing_profile.pk
            }, data={
                'email': getattr(
                    self, '_test_email_address', TEST_EMAIL_ADDRESS
                )
            }
        )
