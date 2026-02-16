from django.contrib.contenttypes.models import ContentType

from mayan.apps.document_states.tests.mixins.workflow_template_state_action_mixins import (
    WorkflowTemplateStateActionTestMixin
)
from mayan.apps.documents.permissions import permission_document_view
from mayan.apps.testing.tests.base import BaseTestCase

from ..workflow_actions import (
    GrantAccessAction, GrantDocumentAccessAction, RevokeAccessAction,
    RevokeDocumentAccessAction
)


class ACLActionTestCase(
    WorkflowTemplateStateActionTestMixin, BaseTestCase
):
    def test_grant_access_action(self):
        self._execute_workflow_template_state_action(
            klass=GrantAccessAction, kwargs={
                'content_type': ContentType.objects.get_for_model(
                    model=self._test_document
                ).pk,
                'object_id': self._test_document.pk,
                'roles': [self._test_case_role.pk],
                'permissions': [permission_document_view.pk]
            }
        )
        self.assertEqual(
            self._test_document.acls.count(), 1
        )
        self.assertEqual(
            list(
                self._test_document.acls.first().permissions.all()
            ), [permission_document_view.stored_permission]
        )
        self.assertEqual(
            self._test_document.acls.first().role, self._test_case_role
        )

    def test_grant_document_access_action(self):
        self._execute_workflow_template_state_action(
            klass=GrantDocumentAccessAction, kwargs={
                'roles': [self._test_case_role.pk],
                'permissions': [permission_document_view.pk]
            }
        )
        self.assertEqual(
            self._test_document.acls.count(), 1
        )
        self.assertEqual(
            list(
                self._test_document.acls.first().permissions.all()
            ), [permission_document_view.stored_permission]
        )
        self.assertEqual(
            self._test_document.acls.first().role, self._test_case_role
        )

    def test_revoke_access_action(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_view
        )

        self._execute_workflow_template_state_action(
            klass=RevokeAccessAction, kwargs={
                'content_type': ContentType.objects.get_for_model(
                    model=self._test_document
                ).pk,
                'object_id': self._test_document.pk,
                'roles': [self._test_case_role.pk],
                'permissions': [permission_document_view.pk]
            }
        )
        self.assertEqual(
            self._test_document.acls.count(), 0
        )

    def test_revoke_document_access_action(self):
        self.grant_access(
            obj=self._test_document, permission=permission_document_view
        )

        self._execute_workflow_template_state_action(
            klass=RevokeDocumentAccessAction, kwargs={
                'roles': [self._test_case_role.pk],
                'permissions': [permission_document_view.pk]
            }
        )
        self.assertEqual(
            self._test_document.acls.count(), 0
        )
