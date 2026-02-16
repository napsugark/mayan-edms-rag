from django.test import tag

from ...models.index_instance_models import IndexInstance

from .index_template_mixins import IndexTemplateTestMixin


@tag('document_indexing')
class IndexInstanceTestMixin(IndexTemplateTestMixin):
    def setUp(self):
        super().setUp()
        self._test_index_instance = IndexInstance.objects.get(
            pk=self._test_index_template.pk
        )

    def _populate_test_index_instance_node(self):
        self._test_index_instance_root_node = self._test_index_instance.index_instance_root_node
        self._test_index_instance_node = self._test_index_instance_root_node.get_children().first()


class DocumentIndexInstanceAPIViewTestMixin(IndexInstanceTestMixin):
    def _request_test_document_index_instance_list_api_view(self):
        return self.get(
            viewname='rest_api:document-index-list', kwargs={
                'document_id': self._test_document.pk
            }
        )


class DocumentIndexInstanceViewTestMixin(IndexInstanceTestMixin):
    def _request_test_document_index_instance_list_view(self):
        return self.get(
            viewname='indexing:document_index_list', kwargs={
                'document_id': self._test_document.pk
            }
        )


class IndexInstanceAPIViewTestMixin(IndexInstanceTestMixin):
    def _request_test_index_instance_list_api_view(self):
        return self.get(viewname='rest_api:indexinstance-list')

    def _request_test_index_instance_detail_api_view(self):
        return self.get(
            viewname='rest_api:indexinstance-detail', kwargs={
                'index_instance_id': self._test_index_instance.pk
            }
        )


class IndexInstanceNodeAPIViewTestMixin(IndexInstanceTestMixin):
    def _request_test_index_instance_node_children_list_api_view(self):
        return self.get(
            viewname='rest_api:indexinstancenode-children-list', kwargs={
                'index_instance_id': self._test_index_instance.pk,
                'index_instance_node_id': self._test_index_instance_node.pk
            }
        )

    def _request_test_index_instance_node_detail_api_view(self):
        return self.get(
            viewname='rest_api:indexinstancenode-detail', kwargs={
                'index_instance_id': self._test_index_instance.pk,
                'index_instance_node_id': self._test_index_instance_node.pk
            }
        )

    def _request_test_index_instance_node_document_list_api_view(self):
        return self.get(
            viewname='rest_api:indexinstancenode-document-list', kwargs={
                'index_instance_id': self._test_index_instance.pk,
                'index_instance_node_id': self._test_index_instance_node.pk
            }
        )

    def _request_test_index_instance_node_list_api_view(self):
        return self.get(
            viewname='rest_api:indexinstancenode-list', kwargs={
                'index_instance_id': self._test_index_instance.pk
            }
        )


class IndexInstanceViewTestMixin(IndexInstanceTestMixin):
    def _request_test_index_instance_node_view(self, index_instance_node):
        return self.get(
            viewname='indexing:index_instance_node_view', kwargs={
                'index_instance_node_id': index_instance_node.pk
            }
        )
