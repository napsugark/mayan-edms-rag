from mayan.apps.document_states.permissions import (
    permission_workflow_template_edit
)
from mayan.apps.document_states.tests.mixins.workflow_template_state_action_mixins import (
    WorkflowTemplateStateActionTestMixin,
    WorkflowTemplateStateActionViewTestMixin
)
from mayan.apps.testing.tests.base import BaseTestCase, GenericViewTestCase

from ..models import LayerTransformation
from ..workflow_actions import TransformationAddAction

from .literals import (
    TEST_TRANSFORMATION_ROTATE_ARGUMENT, TEST_TRANSFORMATION_ROTATE_NAME
)


class TransformationActionTestCase(
    WorkflowTemplateStateActionTestMixin, BaseTestCase
):
    def setUp(self):
        super().setUp()

        self._test_document.delete()
        self._test_document.delete()

        self._upload_test_document()

    def test_transformation_add_pages_all_action(self):
        transformation_count = LayerTransformation.objects.get_for_object(
            obj=self._test_document.pages.first()
        ).count()

        self._execute_workflow_template_state_action(
            klass=TransformationAddAction, kwargs={
                'pages': '',
                'transformation_class': TEST_TRANSFORMATION_ROTATE_NAME,
                'transformation_arguments': TEST_TRANSFORMATION_ROTATE_ARGUMENT
            }
        )
        self.assertEqual(
            LayerTransformation.objects.get_for_object(
                obj=self._test_document.pages.first()
            ).count(), transformation_count + 1
        )

    def test_transformation_add_pages_first_action(self):
        transformation_count = LayerTransformation.objects.get_for_object(
            obj=self._test_document.pages.first()
        ).count()

        self._execute_workflow_template_state_action(
            klass=TransformationAddAction, kwargs={
                'pages': '1',
                'transformation_class': TEST_TRANSFORMATION_ROTATE_NAME,
                'transformation_arguments': TEST_TRANSFORMATION_ROTATE_ARGUMENT
            }
        )
        self.assertEqual(
            LayerTransformation.objects.get_for_object(
                obj=self._test_document.pages.first()
            ).count(), transformation_count + 1
        )


class TransformationActionViewTestCase(
    WorkflowTemplateStateActionViewTestMixin, GenericViewTestCase
):
    def test_transformation_add_pages_all_action_create_view(self):
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )

        response = self._request_test_workflow_template_state_action_create_post_view(
            backend_path='mayan.apps.converter.workflow_actions.TransformationAddAction',
            extra_data={
                'pages': '',
                'transformation_class': TEST_TRANSFORMATION_ROTATE_NAME,
                'transformation_arguments': TEST_TRANSFORMATION_ROTATE_ARGUMENT
            }
        )
        self.assertEqual(response.status_code, 302)

    def test_transformation_add_pages_first_action_create_view(self):
        self.grant_access(
            obj=self._test_workflow_template,
            permission=permission_workflow_template_edit
        )

        response = self._request_test_workflow_template_state_action_create_post_view(
            backend_path='mayan.apps.converter.workflow_actions.TransformationAddAction',
            extra_data={
                'pages': '1',
                'transformation_class': TEST_TRANSFORMATION_ROTATE_NAME,
                'transformation_arguments': TEST_TRANSFORMATION_ROTATE_ARGUMENT
            }
        )
        self.assertEqual(response.status_code, 302)
