from ...models.index_template_models import IndexTemplateNode

from ..literals import (
    TEST_INDEX_TEMPLATE_NODE_EXPRESSION,
    TEST_INDEX_TEMPLATE_NODE_EXPRESSION_EDITED
)

from .index_template_mixins import IndexTemplateTestMixin


class IndexTemplateNodeAPITestMixin(IndexTemplateTestMixin):
    def _request_test_index_template_node_create_api_view(
        self, extra_data=None
    ):
        data = {
            'expression': TEST_INDEX_TEMPLATE_NODE_EXPRESSION
        }

        if extra_data:
            data.update(extra_data)

        values = list(
            IndexTemplateNode.objects.values_list('pk', flat=True)
        )

        response = self.post(
            viewname='rest_api:indextemplatenode-list', kwargs={
                'index_template_id': self._test_index_template.pk
            }, data=data
        )
        self._test_index_template_node = IndexTemplateNode.objects.exclude(
            pk__in=values
        ).first()

        return response

    def _request_test_index_template_node_delete_api_view(self):
        return self.delete(
            viewname='rest_api:indextemplatenode-detail', kwargs={
                'index_template_id': self._test_index_template.pk,
                'index_template_node_id': self._test_index_template_node.pk
            }
        )

    def _request_test_index_template_node_detail_api_view(self):
        return self.get(
            viewname='rest_api:indextemplatenode-detail', kwargs={
                'index_template_id': self._test_index_template.pk,
                'index_template_node_id': self._test_index_template_node.pk
            }
        )

    def _request_test_index_template_node_edit_via_patch_api_view(
        self, extra_data=None
    ):
        data = {
            'enabled': self._test_index_template_node.enabled,
            'expression': self._test_index_template_node.expression,
            'index': self._test_index_template.pk,
            'link_documents': self._test_index_template_node.link_documents,
            'parent': self._test_index_template_node.parent.pk
        }
        data['expression'] = TEST_INDEX_TEMPLATE_NODE_EXPRESSION_EDITED

        if extra_data:
            data.update(**extra_data)

        return self.patch(
            viewname='rest_api:indextemplatenode-detail', kwargs={
                'index_template_id': self._test_index_template.pk,
                'index_template_node_id': self._test_index_template_node.pk
            }, data=data
        )

    def _request_test_index_template_node_list_api_view(self):
        return self.get(
            viewname='rest_api:indextemplatenode-list', kwargs={
                'index_template_id': self._test_index_template.pk
            }
        )


class IndexTemplateNodeViewTestMixin(IndexTemplateTestMixin):
    def _request_test_index_template_node_create_view(self):
        return self.post(
            viewname='indexing:template_node_create', kwargs={
                'index_template_node_id': self._test_index_template.index_template_root_node.pk
            }, data={
                'expression_template': TEST_INDEX_TEMPLATE_NODE_EXPRESSION,
                'index': self._test_index_template.pk,
                'link_document': True
            }
        )

    def _request_test_index_template_node_delete_view(self):
        return self.post(
            viewname='indexing:template_node_delete', kwargs={
                'index_template_node_id': self._test_index_template_node.pk
            }
        )

    def _request_test_index_template_node_edit_view(self):
        return self.post(
            viewname='indexing:template_node_edit', kwargs={
                'index_template_node_id': self._test_index_template_node.pk
            }, data={
                'expression_template': TEST_INDEX_TEMPLATE_NODE_EXPRESSION_EDITED,
                'index': self._test_index_template.pk
            }
        )

    def _request_test_index_template_node_list_view(self):
        return self.get(
            viewname='indexing:index_template_view', kwargs={
                'index_template_id': self._test_index_template.pk
            }
        )
