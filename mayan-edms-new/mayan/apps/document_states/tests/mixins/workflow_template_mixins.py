from django.test import tag

from mayan.apps.documents.tests.mixins.document_mixins import (
    DocumentTestMixin
)

from ...models.workflow_models import Workflow, WorkflowRuntimeProxy
from ...tasks import (
    task_launch_all_workflow_for, task_launch_all_workflows,
    task_launch_workflow, task_launch_workflow_for
)

from ..literals import (
    TEST_WORKFLOW_TEMPLATE_INTERNAL_NAME, TEST_WORKFLOW_TEMPLATE_LABEL,
    TEST_WORKFLOW_TEMPLATE_LABEL_EDITED
)


@tag('workflows')
class WorkflowTemplateTestMixin(DocumentTestMixin):
    _test_object_model = Workflow
    _test_object_name = '_test_workflow_template'
    auto_add_test_workflow_template_test_document_type = False
    auto_create_test_workflow_template = False

    def setUp(self):
        super().setUp()
        self._test_workflow_runtime_proxy = None
        self._test_workflow_runtime_proxy_list = []
        self._test_workflow_template_state_runtime_proxy = None
        self._test_workflow_template_state_runtime_proxy_list = []
        self._test_workflow_template = None
        self._test_workflow_template_list = []

        if self.auto_create_test_workflow_template:
            self._create_test_workflow_template(
                add_test_document_type=self.auto_add_test_workflow_template_test_document_type
            )

    def _create_test_workflow_template(
        self, add_test_document_type=False, auto_launch=True
    ):
        total_test_workflow_template_count = len(
            self._test_workflow_template_list
        )
        label = '{}_{}'.format(
            TEST_WORKFLOW_TEMPLATE_LABEL, total_test_workflow_template_count
        )
        internal_name = '{}_{}'.format(
            TEST_WORKFLOW_TEMPLATE_INTERNAL_NAME,
            total_test_workflow_template_count
        )

        self._test_workflow_template = Workflow.objects.create(
            auto_launch=auto_launch, label=label,
            internal_name=internal_name
        )
        self._test_workflow_template_list.append(
            self._test_workflow_template
        )
        self._test_workflow_runtime_proxy = WorkflowRuntimeProxy.objects.get(
            pk=self._test_workflow_template.pk
        )
        self._test_workflow_runtime_proxy_list.append(
            self._test_workflow_runtime_proxy
        )

        if add_test_document_type:
            self._test_workflow_template.document_types.add(
                self._test_document_type
            )


class DocumentTypeAddRemoveWorkflowTemplateViewTestMixin(
    WorkflowTemplateTestMixin
):
    def _request_test_document_type_workflow_template_add_remove_get_view(self):
        return self.get(
            viewname='document_states:document_type_workflow_templates',
            kwargs={
                'document_type_id': self._test_document_type.pk
            }
        )

    def _request_test_document_type_workflow_template_add_view(self):
        return self.post(
            viewname='document_states:document_type_workflow_templates',
            kwargs={
                'document_type_id': self._test_document_type.pk
            }, data={
                'available-selection': self._test_workflow_template.pk,
                'available-submit': 'true'
            }
        )

    def _request_test_document_type_workflow_template_remove_view(self):
        return self.post(
            viewname='document_states:document_type_workflow_templates',
            kwargs={
                'document_type_id': self._test_document_type.pk
            }, data={
                'added-selection': self._test_workflow_template.pk,
                'added-submit': 'true'
            }
        )


class WorkflowTemplateTaskTestMixin(WorkflowTemplateTestMixin):
    def _execute_task_launch_all_workflows(self):
        task_launch_all_workflows.apply_async().get()

    def _execute_task_launch_all_workflow_for(self):
        task_launch_all_workflow_for.apply_async(
            kwargs={
                'document_id': self._test_document.pk
            }
        ).get()

    def _execute_task_launch_workflow(self):
        task_launch_workflow.apply_async(
            kwargs={
                'workflow_id': self._test_workflow_template.pk
            }
        ).get()

    def _execute_task_launch_workflow_for(self):
        task_launch_workflow_for.apply_async(
            kwargs={
                'document_id': self._test_document.pk,
                'workflow_id': self._test_workflow_template.pk
            }
        ).get()


class WorkflowTemplateDocumentTypeViewTestMixin(WorkflowTemplateTestMixin):
    def _request_test_workflow_template_document_type_add_remove_get_view(self):
        return self.get(
            viewname='document_states:workflow_template_document_types',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk
            }
        )

    def _request_test_workflow_template_document_type_add_view(self):
        return self.post(
            viewname='document_states:workflow_template_document_types',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk
            }, data={
                'available-selection': self._test_document_type.pk,
                'available-submit': 'true'
            }
        )

    def _request_test_workflow_template_document_type_remove_view(self):
        return self.post(
            viewname='document_states:workflow_template_document_types',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk
            }, data={
                'added-selection': self._test_document_type.pk,
                'added-submit': 'true'
            }
        )


class WorkflowTemplateAPIViewTestMixin(WorkflowTemplateTestMixin):
    def _request_test_workflow_template_create_api_view(
        self, extra_data=None
    ):
        data = {
            'internal_name': TEST_WORKFLOW_TEMPLATE_INTERNAL_NAME,
            'label': TEST_WORKFLOW_TEMPLATE_LABEL,
        }

        if extra_data:
            data.update(extra_data)

        self._test_object_track()

        response = self.post(
            viewname='rest_api:workflow-template-list', data=data
        )

        self._test_object_set()

        return response

    def _request_test_workflow_template_delete_api_view(self):
        return self.delete(
            viewname='rest_api:workflow-template-detail', kwargs={
                'workflow_template_id': self._test_workflow_template.pk
            }
        )

    def _request_test_workflow_template_detail_api_view(self):
        return self.get(
            viewname='rest_api:workflow-template-detail', kwargs={
                'workflow_template_id': self._test_workflow_template.pk
            }
        )

    def _request_test_workflow_template_edit_via_patch_view(self):
        return self.patch(
            viewname='rest_api:workflow-template-detail', kwargs={
                'workflow_template_id': self._test_workflow_template.pk
            }, data={'label': TEST_WORKFLOW_TEMPLATE_LABEL_EDITED}
        )

    def _request_test_workflow_template_edit_via_put_view(self):
        return self.put(
            viewname='rest_api:workflow-template-detail', kwargs={
                'workflow_template_id': self._test_workflow_template.pk
            }, data={
                'internal_name': TEST_WORKFLOW_TEMPLATE_INTERNAL_NAME,
                'label': TEST_WORKFLOW_TEMPLATE_LABEL_EDITED
            }
        )

    def _request_test_workflow_template_image_view_api_view(self):
        return self.get(
            viewname='rest_api:workflow-template-image', kwargs={
                'workflow_template_id': self._test_workflow_template.pk
            }
        )

    def _request_test_workflow_template_list_api_view(self):
        return self.get(viewname='rest_api:workflow-template-list')


class WorkflowTemplateDocumentTypeAPIViewMixin(WorkflowTemplateTestMixin):
    def _request_test_workflow_template_document_type_add_api_view(self):
        return self.post(
            viewname='rest_api:workflow-template-document-type-add',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk
            }, data={
                'document_type_id': self._test_document_type.pk
            }
        )

    def _request_test_workflow_template_document_type_list_api_view(self):
        return self.get(
            viewname='rest_api:workflow-template-document-type-list',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk
            }
        )

    def _request_test_workflow_template_document_type_remove_api_view(self):
        return self.post(
            viewname='rest_api:workflow-template-document-type-remove',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk
            }, data={
                'document_type_id': self._test_document_type.pk
            }
        )


class WorkflowToolViewTestMixin(WorkflowTemplateTestMixin):
    def _request_workflow_launch_view(self):
        return self.post(
            viewname='document_states:tool_launch_workflows'
        )


class WorkflowTemplateViewTestMixin(WorkflowTemplateTestMixin):
    def _request_test_workflow_template_create_view(self):
        data = {
            'internal_name': TEST_WORKFLOW_TEMPLATE_INTERNAL_NAME,
            'label': TEST_WORKFLOW_TEMPLATE_LABEL,
        }

        self._test_object_track()

        response = self.post(
            viewname='document_states:workflow_template_create', data=data
        )

        self._test_object_set()

        return response

    def _request_test_workflow_template_delete_view(self):
        return self.post(
            viewname='document_states:workflow_template_single_delete',
            kwargs={
                'workflow_template_id': self._test_workflow_template.pk
            }
        )

    def _request_test_workflow_template_edit_view(self):
        return self.post(
            viewname='document_states:workflow_template_edit', kwargs={
                'workflow_template_id': self._test_workflow_template.pk
            }, data={
                'auto_launch': True,
                'internal_name': self._test_workflow_template.internal_name,
                'label': TEST_WORKFLOW_TEMPLATE_LABEL_EDITED
            }
        )

    def _request_test_workflow_template_list_view(self):
        return self.get(
            viewname='document_states:workflow_template_list',
        )

    def _request_test_workflow_template_preview_view(self):
        return self.get(
            viewname='document_states:workflow_template_preview', kwargs={
                'workflow_template_id': self._test_workflow_template.pk
            }
        )
