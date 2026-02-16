import logging

from django.apps import apps
from django.utils.translation import gettext_lazy as _

from mayan.apps.acls.models import AccessControlList
from mayan.apps.documents.models.document_models import Document
from mayan.apps.documents.permissions import permission_document_view
from mayan.apps.lock_manager.backends.base import LockingBackend
from mayan.apps.lock_manager.exceptions import LockError
from mayan.apps.templating.template_backends import Template

logger = logging.getLogger(name=__name__)


class IndexInstanceBusinessLogicMixin:
    def _delete_empty_nodes(self):
        self.index_instance_root_node.get_children().filter(
            children=None, documents=None
        ).delete()

    def _document_add(self, document, index_instance_node_parent):
        for index_template_node in index_instance_node_parent.index_template_node.get_children().filter(enabled=True):
            try:
                template = Template(
                    template_string=index_template_node.expression
                )
                result = template.render(
                    context={'document': document}
                )
            except Exception as exception:
                logger.error('Evaluating error: %s', exception)
                error_message = _(
                    message='Error indexing document: %(document)s; expression: '
                    '%(expression)s; %(exception)s'
                ) % {
                    'document': document,
                    'exception': exception,
                    'expression': index_template_node.expression
                }
                logger.debug(msg=error_message)
            else:
                logger.debug('Evaluation result: %s', result)

                if result:
                    index_instance_node, created = index_template_node.index_instance_nodes.get_or_create(
                        parent=index_instance_node_parent,
                        value=result
                    )

                    if index_template_node.link_documents:
                        index_instance_node.documents.add(document)

                    self._document_add(
                        document=document,
                        index_instance_node_parent=index_instance_node
                    )

    def document_nodes_delete(self, document):
        IndexInstanceNode = apps.get_model(
            app_label='document_indexing', model_name='IndexInstanceNode'
        )

        IndexInstanceNode.documents.through.objects.filter(
            document=document,
            indexinstancenode__index_template_node__enabled=True,
            indexinstancenode__index_template_node__index=self,
            indexinstancenode__index_template_node__index__enabled=True,
            indexinstancenode__index_template_node__index__document_types=document.document_type
        ).delete()

    def document_add(self, document):
        """
        Method to start the indexing process for a document. The entire
        process happens inside one transaction. The document is first
        removed from all the index nodes to which it already belongs.
        The different index templates that match this document's type
        are evaluated and for each result a node is fetched or created and
        the document is added to that node.
        """
        logger.debug('Index; Indexing document: %s', document)

        if Document.valid.filter(pk=document.pk).exists() and self.enabled and self.document_types.filter(pk=document.document_type.pk).exists():
            try:
                locking_backend = LockingBackend.get_backend()

                lock_index_instance = locking_backend.acquire_lock(
                    name=self.get_lock_string()
                )
            except LockError:
                raise
            else:
                try:
                    lock_document = locking_backend.acquire_lock(
                        name=self.get_document_lock_string(document=document)
                    )
                except LockError:
                    raise
                else:
                    try:
                        self.initialize_index_instance_root_node_node()

                        self.document_nodes_delete(document=document)

                        self._document_add(
                            document=document,
                            index_instance_node_parent=self.index_instance_root_node
                        )

                        self._delete_empty_nodes()
                    finally:
                        lock_document.release()
                finally:
                    lock_index_instance.release()

    def document_remove(self, document):
        if self.enabled and self.document_types.filter(pk=document.document_type.pk).exists():
            try:
                lock_index_instance = LockingBackend.get_backend().acquire_lock(
                    name=self.get_lock_string()
                )
            except LockError:
                raise
            else:
                try:
                    lock_document = LockingBackend.get_backend().acquire_lock(
                        name=self.get_document_lock_string(
                            document=document
                        )
                    )
                except LockError:
                    raise
                else:
                    try:
                        self.document_nodes_delete(document=document)
                        self._delete_empty_nodes()
                    finally:
                        lock_document.release()
                finally:
                    lock_index_instance.release()

    def get_children(self):
        return self.index_instance_root_node.get_children()

    def get_document_lock_string(self, document):
        return 'indexing:document_{}'.format(document.pk)

    def get_descendants(self):
        return self.index_instance_root_node.get_descendants()

    def get_descendants_count(self):
        return self.index_instance_root_node.get_descendants_count()

    get_descendants_count.help_text = _(
        message='Total number of nodes with unique values this item contains.'
    )

    def get_descendants_document_count(self, user):
        return self.index_instance_root_node.get_descendants_document_count(
            user=user
        )

    get_descendants_document_count.help_text = _(
        message='Total number of unique documents this item contains.'
    )

    def get_lock_string(self):
        return 'indexing:index_instance_{}'.format(self.pk)

    def get_level_count(self):
        return self.index_instance_root_node.get_level_count()

    get_level_count.help_text = _(
        message='Total number of node levels this item contains.'
    )

    def get_root(self):
        """Compatibility method."""
        return self.index_instance_root_node

    @property
    def index_instance_root_node(self):
        return self.index_template_root_node.get_index_instance_root_node()

    def initialize_index_instance_root_node_node(self):
        return self.index_template_root_node.initialize_index_instance_root_node()


class IndexInstanceNodeBusinessLogicMixin:
    def _get_documents(self):
        return Document.valid.filter(
            pk__in=self.documents.values('pk')
        )

    def get_children_count(self):
        return self.get_children().count()

    def get_descendants_count(self):
        return self.get_descendants().count()

    get_descendants_count.help_text = IndexInstanceBusinessLogicMixin.get_descendants_count.help_text

    def get_descendants_document_count(self, user):
        queryset = Document.valid.filter(
            index_instance_nodes__in=self.get_descendants(
                include_self=True
            )
        ).distinct()

        return AccessControlList.objects.restrict_queryset(
            permission=permission_document_view,
            queryset=queryset, user=user
        ).count()

    get_descendants_document_count.help_text = IndexInstanceBusinessLogicMixin.get_descendants_document_count.help_text

    def get_documents(self, permission, user):
        """
        Provide a queryset of the documents in an index instance node.
        The queryset is filtered by access.
        """
        return AccessControlList.objects.restrict_queryset(
            permission=permission, queryset=self._get_documents(),
            user=user
        )

    def get_full_path(self):
        result = []
        for node in self.get_ancestors(include_self=True):
            if node.is_root_node():
                result.append(
                    str(
                        self.index()
                    )
                )
            else:
                result.append(
                    str(node)
                )

        return ' / '.join(result)
    get_full_path.help_text = _(
        message='The path to the index including all ancestors.'
    )
    get_full_path.short_description = _(message='Full path')

    def get_level_count(self):
        return self.get_descendants().values('level').distinct().count()

    get_level_count.help_text = IndexInstanceBusinessLogicMixin.get_level_count.help_text

    def index(self):
        """
        Return's the index instance of this node instance.
        """
        IndexInstance = apps.get_model(
            app_label='document_indexing', model_name='IndexInstance'
        )

        return IndexInstance.objects.get(
            pk=self.index_template_node.index.pk
        )
