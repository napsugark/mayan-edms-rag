import json

from django.core import mail

from mayan.apps.document_states.events import (
    event_workflow_instance_created, event_workflow_template_edited
)
from mayan.apps.document_states.literals import WORKFLOW_ACTION_ON_ENTRY
from mayan.apps.document_states.permissions import (
    permission_workflow_template_edit
)
from mayan.apps.document_states.tests.mixins.workflow_template_state_action_mixins import (
    WorkflowTemplateStateActionTestMixin,
    WorkflowTemplateStateActionViewTestMixin
)
from mayan.apps.documents.events import event_document_created
from mayan.apps.metadata.tests.mixins.metadata_type_mixins import (
    MetadataTypeTestMixin
)
from mayan.apps.testing.tests.base import BaseTestCase, GenericViewTestCase

from ..events import event_email_sent
from ..permissions import permission_mailing_profile_use
from ..workflow_actions import DocumentEmailAction

from .literals import (
    TEST_EMAIL_ADDRESS, TEST_EMAIL_BODY, TEST_EMAIL_FROM_ADDRESS,
    TEST_EMAIL_SUBJECT
)
from .mixins import MailingProfileTestMixin


class DocumentEmailActionTestCase(
    MailingProfileTestMixin, WorkflowTemplateStateActionTestMixin,
    BaseTestCase
):
    def test_email_action_literal_text(self):
        self._create_test_mailing_profile()

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=DocumentEmailAction, kwargs={
                'mailing_profile': self._test_mailing_profile.pk,
                'recipient': TEST_EMAIL_ADDRESS,
                'subject': TEST_EMAIL_SUBJECT,
                'body': TEST_EMAIL_BODY
            }
        )
        self.assertEqual(
            len(mail.outbox), 1
        )
        self.assertEqual(
            mail.outbox[0].from_email, TEST_EMAIL_FROM_ADDRESS
        )
        self.assertEqual(
            mail.outbox[0].to, [TEST_EMAIL_ADDRESS]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, self._test_mailing_profile)
        self.assertEqual(events[0].target, self._test_mailing_profile)
        self.assertEqual(events[0].verb, event_email_sent.id)

    def test_email_action_literal_text_cc_field(self):
        self._create_test_mailing_profile()

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=DocumentEmailAction, kwargs={
                'mailing_profile': self._test_mailing_profile.pk,
                'recipient': TEST_EMAIL_ADDRESS,
                'cc': TEST_EMAIL_ADDRESS,
                'subject': TEST_EMAIL_SUBJECT,
                'body': TEST_EMAIL_BODY
            }
        )
        self.assertEqual(
            len(mail.outbox), 1
        )
        self.assertEqual(
            mail.outbox[0].from_email, TEST_EMAIL_FROM_ADDRESS
        )
        self.assertEqual(
            mail.outbox[0].to, [TEST_EMAIL_ADDRESS]
        )
        self.assertEqual(
            mail.outbox[0].cc, [TEST_EMAIL_ADDRESS]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, self._test_mailing_profile)
        self.assertEqual(events[0].target, self._test_mailing_profile)
        self.assertEqual(events[0].verb, event_email_sent.id)

    def test_email_action_literal_text_bcc_field(self):
        self._create_test_mailing_profile()

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=DocumentEmailAction, kwargs={
                'mailing_profile': self._test_mailing_profile.pk,
                'recipient': TEST_EMAIL_ADDRESS,
                'bcc': TEST_EMAIL_ADDRESS,
                'subject': TEST_EMAIL_SUBJECT,
                'body': TEST_EMAIL_BODY
            }
        )
        self.assertEqual(
            len(mail.outbox), 1
        )
        self.assertEqual(
            mail.outbox[0].from_email, TEST_EMAIL_FROM_ADDRESS
        )
        self.assertEqual(
            mail.outbox[0].to, [TEST_EMAIL_ADDRESS]
        )
        self.assertEqual(
            mail.outbox[0].bcc, [TEST_EMAIL_ADDRESS]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, self._test_mailing_profile)
        self.assertEqual(events[0].target, self._test_mailing_profile)
        self.assertEqual(events[0].verb, event_email_sent.id)

    def test_email_action_workflow_execute(self):
        self._create_test_mailing_profile()

        self._test_workflow_template_state.actions.create(
            backend_data=json.dumps(
                obj={
                    'mailing_profile': self._test_mailing_profile.pk,
                    'recipient': TEST_EMAIL_ADDRESS,
                    'subject': TEST_EMAIL_SUBJECT,
                    'body': TEST_EMAIL_BODY
                }
            ),
            backend_path='mayan.apps.mailer.workflow_actions.DocumentEmailAction',
            label='test email action', when=WORKFLOW_ACTION_ON_ENTRY
        )

        self._test_workflow_template_state.initial = True
        self._test_workflow_template_state.save()
        self._test_workflow_template.document_types.add(
            self._test_document_type
        )

        self._clear_events()

        self._create_test_document_stub()

        self.assertEqual(
            len(mail.outbox), 1
        )
        self.assertEqual(
            mail.outbox[0].from_email, TEST_EMAIL_FROM_ADDRESS
        )
        self.assertEqual(
            mail.outbox[0].to, [TEST_EMAIL_ADDRESS]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 3)

        self.assertEqual(events[0].action_object, self._test_document_type)
        self.assertEqual(events[0].actor, self._test_document)
        self.assertEqual(events[0].target, self._test_document)
        self.assertEqual(events[0].verb, event_document_created.id)

        self.assertEqual(events[1].action_object, self._test_document)
        self.assertEqual(events[1].actor, self._test_workflow_instance)
        self.assertEqual(events[1].target, self._test_workflow_instance)
        self.assertEqual(events[1].verb, event_workflow_instance_created.id)

        self.assertEqual(events[2].action_object, self._test_document)
        self.assertEqual(events[2].actor, self._test_mailing_profile)
        self.assertEqual(events[2].target, self._test_mailing_profile)
        self.assertEqual(events[2].verb, event_email_sent.id)


class DocumentEmailActionTemplateTestCase(
    MetadataTypeTestMixin, MailingProfileTestMixin,
    WorkflowTemplateStateActionTestMixin, BaseTestCase
):
    def test_email_action_recipient_template(self):
        self._create_test_metadata_type()
        self._test_document_type.metadata.create(
            metadata_type=self._test_metadata_type
        )
        self._test_document.metadata.create(
            metadata_type=self._test_metadata_type, value=TEST_EMAIL_ADDRESS
        )

        self._create_test_mailing_profile()

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=DocumentEmailAction, kwargs={
                'mailing_profile': self._test_mailing_profile.pk,
                'recipient': '{{{{ workflow_instance.document.metadata_value_of.{} }}}}'.format(
                    self._test_metadata_type.name
                ),
                'subject': TEST_EMAIL_SUBJECT,
                'body': ''
            }
        )
        self.assertEqual(
            len(mail.outbox), 1
        )
        self.assertEqual(
            mail.outbox[0].from_email, TEST_EMAIL_FROM_ADDRESS
        )
        self.assertEqual(
            len(mail.outbox), 1
        )
        self.assertEqual(
            mail.outbox[0].from_email, TEST_EMAIL_FROM_ADDRESS
        )
        self.assertEqual(
            mail.outbox[0].to, [
                self._test_document.metadata.first().value
            ]
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, self._test_mailing_profile)
        self.assertEqual(events[0].target, self._test_mailing_profile)
        self.assertEqual(events[0].verb, event_email_sent.id)

    def test_email_action_subject_template(self):
        self._create_test_metadata_type()
        self._test_document_type.metadata.create(
            metadata_type=self._test_metadata_type
        )
        self._test_document.metadata.create(
            metadata_type=self._test_metadata_type, value=TEST_EMAIL_SUBJECT
        )

        self._create_test_mailing_profile()

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=DocumentEmailAction, kwargs={
                'mailing_profile': self._test_mailing_profile.pk,
                'recipient': TEST_EMAIL_ADDRESS,
                'subject': '{{{{ workflow_instance.document.metadata_value_of.{} }}}}'.format(
                    self._test_metadata_type.name
                ),
                'body': ''
            }
        )
        self.assertEqual(
            len(mail.outbox), 1
        )
        self.assertEqual(
            mail.outbox[0].from_email, TEST_EMAIL_FROM_ADDRESS
        )
        self.assertEqual(
            mail.outbox[0].to, [TEST_EMAIL_ADDRESS]
        )
        self.assertEqual(
            mail.outbox[0].subject,
            self._test_document.metadata.first().value
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, self._test_mailing_profile)
        self.assertEqual(events[0].target, self._test_mailing_profile)
        self.assertEqual(events[0].verb, event_email_sent.id)

    def test_email_action_body_template(self):
        self._create_test_metadata_type()
        self._test_document_type.metadata.create(
            metadata_type=self._test_metadata_type
        )
        self._test_document.metadata.create(
            metadata_type=self._test_metadata_type, value=TEST_EMAIL_BODY
        )

        self._create_test_mailing_profile()

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=DocumentEmailAction, kwargs={
                'mailing_profile': self._test_mailing_profile.pk,
                'recipient': TEST_EMAIL_ADDRESS,
                'subject': TEST_EMAIL_SUBJECT,
                'body': '{{{{ workflow_instance.document.metadata_value_of.{} }}}}'.format(
                    self._test_metadata_type.name
                )
            }
        )
        self.assertEqual(
            len(mail.outbox), 1
        )
        self.assertEqual(
            mail.outbox[0].from_email, TEST_EMAIL_FROM_ADDRESS
        )
        self.assertEqual(
            mail.outbox[0].to, [TEST_EMAIL_ADDRESS]
        )
        self.assertEqual(
            mail.outbox[0].body, TEST_EMAIL_BODY
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_document)
        self.assertEqual(events[0].actor, self._test_mailing_profile)
        self.assertEqual(events[0].target, self._test_mailing_profile)
        self.assertEqual(events[0].verb, event_email_sent.id)

    def test_email_action_attachment(self):
        # This action requires a document with an active version.
        self._upload_test_document()

        self._create_test_metadata_type()
        self._test_document_type.metadata.create(
            metadata_type=self._test_metadata_type
        )
        self._test_document.metadata.create(
            metadata_type=self._test_metadata_type, value=TEST_EMAIL_SUBJECT
        )

        self._create_test_mailing_profile()

        self._clear_events()

        self._execute_workflow_template_state_action(
            klass=DocumentEmailAction, kwargs={
                'mailing_profile': self._test_mailing_profile.pk,
                'recipient': TEST_EMAIL_ADDRESS,
                'subject': '{{{{ workflow_instance.document.metadata_value_of.{} }}}}'.format(
                    self._test_metadata_type.name
                ),
                'body': '',
                'attachment': True
            }
        )
        self.assertEqual(
            len(mail.outbox), 1
        )
        self.assertEqual(
            mail.outbox[0].from_email, TEST_EMAIL_FROM_ADDRESS
        )
        self.assertEqual(
            mail.outbox[0].to, [TEST_EMAIL_ADDRESS]
        )
        self.assertEqual(
            mail.outbox[0].subject,
            self._test_document.metadata.first().value
        )
        self.assertEqual(
            len(mail.outbox[0].attachments), 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(events[0].action_object, self._test_document_version)
        self.assertEqual(events[0].actor, self._test_mailing_profile)
        self.assertEqual(events[0].target, self._test_mailing_profile)
        self.assertEqual(events[0].verb, event_email_sent.id)


class DocumentEmailActionViewTestCase(
    WorkflowTemplateStateActionViewTestMixin, MailingProfileTestMixin,
    GenericViewTestCase
):
    """
    `WorkflowTemplateStateActionViewTestMixin` must precede
    `MailingProfileTestMixin` to allow `self._test_object_track` to work for
    the right model.
    """
    auto_upload_test_document = False
    auto_create_test_workflow_template_state_action = False

    def test_email_action_create_get_view(self):
        self._create_test_mailing_profile()
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )

        self._clear_events()

        response = self._request_test_workflow_template_state_action_create_get_view(
            backend_path='mayan.apps.mailer.workflow_actions.DocumentEmailAction'
        )
        self.assertEqual(response.status_code, 200)

        self.assertEqual(
            self._test_workflow_template_state.actions.count(), 0
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_email_action_create_post_view(self):
        self._create_test_mailing_profile()
        self.grant_access(
            obj=self._test_mailing_profile,
            permission=permission_mailing_profile_use
        )
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )

        self._clear_events()

        response = self._request_test_workflow_template_state_action_create_post_view(
            backend_path='mayan.apps.mailer.workflow_actions.DocumentEmailAction',
            extra_data={
                'mailing_profile': self._test_mailing_profile.pk,
                'recipient': TEST_EMAIL_ADDRESS,
                'subject': TEST_EMAIL_SUBJECT,
                'body': TEST_EMAIL_BODY
            }
        )
        self.assertEqual(response.status_code, 302)

        self.assertEqual(
            self._test_workflow_template_state.actions.count(), 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 1)

        self.assertEqual(
            events[0].action_object,
            self._test_workflow_template_state_action
        )
        self.assertEqual(events[0].actor, self._test_case_user)
        self.assertEqual(events[0].target, self._test_workflow_template)
        self.assertEqual(events[0].verb, event_workflow_template_edited.id)
