import hashlib
import logging

from furl import furl
from graphviz import Digraph

from django.apps import apps
from django.core import serializers
from django.db import IntegrityError
from django.urls import reverse
from django.utils.functional import cached_property
from django.utils.translation import gettext_lazy as _

from mayan.apps.acls.models import AccessControlList
from mayan.apps.documents.models.document_models import Document
from mayan.apps.documents.permissions import permission_document_view
from mayan.apps.file_caching.models import CachePartitionFile

from ..events import event_workflow_template_edited
from ..literals import (
    ERROR_LOG_DOMAIN_NAME, GRAPHVIZ_RANKDIR, GRAPHVIZ_RANKSEP,
    GRAPHVIZ_SPLINES, STORAGE_NAME_WORKFLOW_CACHE
)

logger = logging.getLogger(name=__name__)


class WorkflowBusinessLogicMixin:
    @cached_property
    def cache(self):
        Cache = apps.get_model(app_label='file_caching', model_name='Cache')

        return Cache.objects.get(
            defined_storage_name=STORAGE_NAME_WORKFLOW_CACHE
        )

    @cached_property
    def cache_partition(self):
        partition, created = self.cache.partitions.get_or_create(
            name='{}'.format(self.pk)
        )
        return partition

    def do_diagram_generate(self):
        diagram = Digraph(
            name='finite_state_machine', graph_attr={
                'rankdir': GRAPHVIZ_RANKDIR, 'ranksep': GRAPHVIZ_RANKSEP,
                'splines': GRAPHVIZ_SPLINES
            }, format='png'
        )

        for state in self.states.order_by('completion', 'label'):
            state.do_diagram_generate(diagram=diagram)

        for transition in self.transitions.all():
            transition.do_diagram_generate(diagram=diagram)

        return diagram.pipe()

    def document_types_add(self, queryset, user):
        for model_instance in queryset.all():
            self.document_types.add(model_instance)
            event_workflow_template_edited.commit(
                action_object=model_instance, actor=user, target=self
            )

    def document_types_remove(self, queryset, user):
        for model_instance in queryset.all():
            self.document_types.remove(model_instance)
            event_workflow_template_edited.commit(
                action_object=model_instance, actor=user, target=self
            )
            self.instances.filter(
                document__document_type_id=model_instance.pk
            ).delete()

    def generate_image(
        self, maximum_layer_order=None, transformation_instance_list=None,
        user=None
    ):
        # `user` argument added for compatibility.
        cache_filename = '{}'.format(
            self.get_hash()
        )

        try:
            self.cache_partition.get_file(filename=cache_filename)
        except CachePartitionFile.DoesNotExist:
            logger.debug(
                'workflow cache file "%s" not found', cache_filename
            )

            image = self.do_diagram_generate()
            with self.cache_partition.create_file(filename=cache_filename) as file_object:
                file_object.write(image)
        else:
            logger.debug(
                'workflow cache file "%s" found', cache_filename
            )

        return cache_filename

    def get_api_image_url(self, *args, **kwargs):
        final_url = furl()
        final_url.args = kwargs
        final_url.path = reverse(
            kwargs={'workflow_template_id': self.pk},
            viewname='rest_api:workflow-template-image'
        )
        final_url.args['_hash'] = self.get_hash()

        return final_url.tostr()

    def get_document_types_not_in_workflow(self):
        DocumentType = apps.get_model(
            app_label='documents', model_name='DocumentType'
        )

        return DocumentType.objects.exclude(
            pk__in=self.document_types.all()
        )

    def get_hash(self):
        result = hashlib.sha256(
            string=serializers.serialize(
                format='json', queryset=(self,)
            ).encode()
        )
        for state in self.states.all():
            result.update(
                state.get_hash().encode()
            )

        for transition in self.transitions.all():
            result.update(
                transition.get_hash().encode()
            )

        return result.hexdigest()

    def get_state_final(self):
        try:
            return self.states.get(final=True)
        except self.states.model.DoesNotExist:
            return None
    get_state_final.short_description = _(message='Final state')

    def get_state_initial(self):
        try:
            return self.states.get(initial=True)
        except self.states.model.DoesNotExist:
            return None
    get_state_initial.short_description = _(message='Initial state')

    def launch_for(self, document, user=None):
        WorkflowInstance = apps.get_model(
            app_label='document_states',
            model_name='WorkflowInstance'
        )

        initial_state = self.get_state_initial()

        if initial_state:
            queryset = self.document_types.all()
            if queryset.filter(pk=document.document_type.pk).exists():
                try:
                    logger.debug(
                        'Launching workflow %s for document %s', self, document
                    )
                    workflow_instance = WorkflowInstance(
                        document=document, workflow=self
                    )
                    workflow_instance._event_actor = user
                    workflow_instance.save()

                    initial_state.do_active_set(
                        workflow_instance=workflow_instance
                    )
                    # TODO: Update once initial entry log patch is merged.
                    # Break pattern by passing `workflow_instance`
                    # until initial entry logs patch is merged.
                except IntegrityError:
                    logger.debug(
                        'Workflow %s already launched for document %s',
                        self, document
                    )
                else:
                    logger.debug(
                        'Workflow %s launched for document %s', self, document
                    )
                    return workflow_instance
            else:
                logger.error(
                    'This workflow is not valid for the document type of the '
                    'document.'
                )
        else:
            text_error = '''
            Cannot create a workflow instance. The workflow template `{}`
            does not have an initial state.
            '''.format(self)

            document.error_log.create(
                domain_name=ERROR_LOG_DOMAIN_NAME, text=text_error
            )


class WorkflowRuntimeProxyBusinessLogicMixin:
    def get_documents(self, permission=None, user=None):
        """
        Provide a queryset of the documents. The queryset is optionally
        filtered by access.
        """
        queryset = Document.valid.filter(workflows__workflow=self)

        if self.ignore_completed:
            queryset = queryset.exclude(workflows__state_active__final=True)

        if permission and user:
            queryset = AccessControlList.objects.restrict_queryset(
                permission=permission, queryset=queryset,
                user=user
            )

        return queryset

    def get_document_count(self, user):
        """
        Return the numeric count of documents executing this workflow.
        The count is filtered by access.
        """
        return self.get_documents(
            permission=permission_document_view, user=user
        ).count()

    def get_states(self):
        WorkflowStateRuntimeProxy = apps.get_model(
            app_label='document_states',
            model_name='WorkflowStateRuntimeProxy'
        )

        queryset = WorkflowStateRuntimeProxy.objects.filter(workflow=self)

        if self.ignore_completed:
            queryset = queryset.exclude(final=True)

        return queryset
