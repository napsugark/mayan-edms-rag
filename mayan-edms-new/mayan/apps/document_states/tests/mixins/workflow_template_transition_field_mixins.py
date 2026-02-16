from ...models.workflow_transition_field_models import WorkflowTransitionField

from ..literals import (
    TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_HELP_TEXT,
    TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_LABEL,
    TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_LABEL_EDITED,
    TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_NAME,
    TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_TYPE
)

from .workflow_template_transition_mixins import (
    WorkflowTemplateTransitionTestMixin
)


class WorkflowTemplateTransitionFieldTestMixin(
    WorkflowTemplateTransitionTestMixin
):
    _test_object_model = WorkflowTransitionField
    _test_object_name = '_test_workflow_template_transition_field'

    def _create_test_workflow_template_transition_field(
        self, extra_data=None
    ):
        kwargs = {
            'field_type': TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_TYPE,
            'help_text': TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_HELP_TEXT,
            'label': TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_LABEL,
            'name': TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_NAME
        }
        kwargs.update(
            extra_data or {}
        )

        self._test_workflow_template_transition_field = self._test_workflow_template_transition.fields.create(
            **kwargs
        )


class WorkflowTemplateTransitionFieldAPIViewTestMixin(
    WorkflowTemplateTransitionFieldTestMixin
):
    def _request_test_workflow_template_transition_field_create_api_view(self):
        self._test_object_track()

        response = self.post(
            viewname='rest_api:workflow-template-transition-field-list',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk,
                'workflow_template_transition_id': self._test_workflow_template_transition.pk
            }, data={
                'field_type': TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_TYPE,
                'help_text': TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_HELP_TEXT,
                'label': TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_LABEL,
                'name': TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_NAME
            }
        )

        self._test_object_set()

        return response

    def _request_test_workflow_template_transition_field_delete_api_view(self):
        return self.delete(
            viewname='rest_api:workflow-template-transition-field-detail',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk,
                'workflow_template_transition_id': self._test_workflow_template_transition.pk,
                'workflow_template_transition_field_id': self._test_workflow_template_transition_field.pk
            }
        )

    def _request_test_workflow_template_transition_field_detail_api_view(self):
        return self.get(
            viewname='rest_api:workflow-template-transition-field-detail',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk,
                'workflow_template_transition_id': self._test_workflow_template_transition.pk,
                'workflow_template_transition_field_id': self._test_workflow_template_transition_field.pk
            }
        )

    def _request_test_workflow_template_transition_field_edit_via_patch_api_view(self):
        return self.patch(
            viewname='rest_api:workflow-template-transition-field-detail',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk,
                'workflow_template_transition_id': self._test_workflow_template_transition.pk,
                'workflow_template_transition_field_id': self._test_workflow_template_transition_field.pk
            }, data={
                'label': '{} edited'.format(
                    self._test_workflow_template_transition_field
                )
            }
        )

    def _request_test_workflow_template_transition_field_list_api_view(self):
        return self.get(
            viewname='rest_api:workflow-template-transition-field-list',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk,
                'workflow_template_transition_id': self._test_workflow_template_transition.pk
            }
        )


class WorkflowTemplateTransitionFieldViewTestMixin(
    WorkflowTemplateTransitionFieldTestMixin
):
    def _request_workflow_template_transition_field_create_view(
        self, extra_data=None
    ):
        self._test_object_track()

        data = {
            'field_type': TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_TYPE,
            'help_text': TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_HELP_TEXT,
            'label': TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_LABEL,
            'name': TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_NAME
        }

        if extra_data:
            data.update(extra_data)

        response = self.post(
            viewname='document_states:workflow_template_transition_field_create',
            kwargs={
                'workflow_template_transition_id': self._test_workflow_template_transition.pk
            }, data=data
        )

        self._test_object_set()

        return response

    def _request_workflow_template_transition_field_delete_view(self):
        return self.post(
            viewname='document_states:workflow_template_transition_field_delete',
            kwargs={
                'workflow_template_transition_field_id': self._test_workflow_template_transition_field.pk
            }
        )

    def _request_workflow_template_transition_field_edit_view(self):
        return self.post(
            viewname='document_states:workflow_template_transition_field_edit',
            kwargs={
                'workflow_template_transition_field_id': self._test_workflow_template_transition_field.pk
            }, data={
                'field_type': TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_TYPE,
                'help_text': TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_HELP_TEXT,
                'label': TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_LABEL_EDITED,
                'name': TEST_WORKFLOW_TEMPLATE_TRANSITION_FIELD_NAME
            }
        )

    def _request_test_workflow_template_transition_field_list_view(self):
        return self.get(
            viewname='document_states:workflow_template_transition_field_list',
            kwargs={
                'workflow_template_transition_id': self._test_workflow_template_transition.pk
            }
        )
