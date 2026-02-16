import logging

from django.core.exceptions import ValidationError
from django.db.utils import IntegrityError
from django.utils.translation import gettext_lazy as _

from mayan.apps.document_states.classes import WorkflowAction
from mayan.apps.document_states.exceptions import WorkflowStateActionError
from mayan.apps.document_states.models.workflow_instance_models import (
    WorkflowInstance
)

from .models.metadata_instance_models import DocumentMetadata
from .models.metadata_type_models import MetadataType
from .permissions import (
    permission_document_metadata_add, permission_document_metadata_edit,
    permission_document_metadata_remove
)

logger = logging.getLogger(name=__name__)


class DocumentMetadataAddAction(WorkflowAction):
    form_field_widgets = {
        'metadata_types': {
            'class': 'django.forms.widgets.SelectMultiple', 'kwargs': {
                'attrs': {'class': 'select2'}
            }
        }
    }
    label = _(message='Add metadata')
    permission = permission_document_metadata_add

    @classmethod
    def get_form_fields(cls):
        fields = super().get_form_fields()

        document_types_queryset = cls.workflow_template_state.workflow.document_types

        metadata_type_queryset = MetadataType.objects.get_for_document_types(
            queryset=document_types_queryset
        )

        fields.update(
            {
                'metadata_types': {
                    'class': 'mayan.apps.forms.form_fields.FormFieldFilteredModelChoiceMultiple',
                    'help_text': _(
                        message='Metadata types to add to the document.'
                    ),
                    'kwargs': {
                        'source_queryset': metadata_type_queryset,
                        'permission': cls.permission
                    },
                    'label': _(message='Metadata types'),
                    'required': True
                }
            }
        )

        return fields

    @classmethod
    def get_form_fieldsets(cls):
        fieldsets = super().get_form_fieldsets()

        fieldsets += (
            (
                _(message='Metadata types'), {
                    'fields': ('metadata_types',)
                },
            ),
        )
        return fieldsets

    def execute(self, context):
        for metadata_type in self.get_metadata_types():
            try:
                context['workflow_instance'].document.metadata.create(metadata_type=metadata_type)
            except IntegrityError as exception:
                """This document already has the metadata type added."""
                raise WorkflowStateActionError(
                    _(
                        message='Unable to add metadata type "%(metadata_type)s" '
                        'from document: %(document)s. Exception: '
                        '%(exception)s'
                    ) % {
                        'document': context['workflow_instance'].document,
                        'exception': exception,
                        'metadata_type': metadata_type
                    }
                ) from exception

    def get_metadata_types(self):
        return MetadataType.objects.filter(
            pk__in=self.kwargs.get(
                'metadata_types', ()
            )
        )


class DocumentMetadataEditAction(WorkflowAction):
    form_field_widgets = {
        'metadata_type': {
            'class': 'django.forms.widgets.Select', 'kwargs': {
                'attrs': {'class': 'select2'}
            }
        }
    }
    form_fields = {
        'value': {
            'label': _(message='Value'),
            'class': 'mayan.apps.templating.fields.ModelTemplateField',
            'kwargs': {
                'initial_help_text': _(
                    message='Value to assign to the metadata. '
                    'Can be a literal value or template code.'
                ),
                'model': WorkflowInstance,
                'model_variable': 'workflow_instance',
                'required': True
            }
        }
    }
    label = _(message='Edit metadata')

    @classmethod
    def get_form_fields(cls):
        fields = super().get_form_fields()

        document_types_queryset = cls.workflow_template_state.workflow.document_types

        metadata_type_queryset = MetadataType.objects.get_for_document_types(
            queryset=document_types_queryset
        )

        fields.update(
            {
                'metadata_type': {
                    'class': 'mayan.apps.forms.form_fields.FormFieldFilteredModelChoice',
                    'help_text': _(
                        message='Metadata types to edit.'
                    ),
                    'kwargs': {
                        'source_queryset': metadata_type_queryset,
                        'permission': permission_document_metadata_edit
                    },
                    'label': _(message='Metadata type'),
                    'required': True
                }
            }
        )

        return fields

    @classmethod
    def get_form_fieldsets(cls):
        fieldsets = super().get_form_fieldsets()

        fieldsets += (
            (
                _(message='Metadata'), {
                    'fields': ('metadata_type', 'value')
                },
            ),
        )
        return fieldsets

    def execute(self, context):
        try:
            metadata_type = self.get_metadata_type()
            document_metadata = context['workflow_instance'].document.metadata.get(
                metadata_type=metadata_type
            )
        except DocumentMetadata.DoesNotExist as exception:
            """
            Non fatal, we just ignore the action to edit the metadata value.
            """
            raise WorkflowStateActionError(
                _(
                    message='Unable to edit metadata type "%(metadata_type)s" '
                    'from document: %(document)s. Document does not have '
                    'the metadata type to be edited. Exception: '
                    '%(exception)s'
                ) % {
                    'document': context['workflow_instance'].document,
                    'exception': exception,
                    'metadata_type': metadata_type
                }
            ) from exception
        else:
            document_metadata.value = self.render_field(
                field_name='value', context=context
            )
            document_metadata.save()

    def get_metadata_type(self):
        return MetadataType.objects.get(
            pk=self.kwargs['metadata_type']
        )


class DocumentMetadataRemoveAction(DocumentMetadataAddAction):
    label = _(message='Remove metadata')
    permission = permission_document_metadata_remove

    @classmethod
    def get_form_fields(cls):
        fields = super().get_form_fields()

        fields['metadata_types']['help_text'] = _(
            message='Metadata types to remove from the document.'
        )

        return fields

    def execute(self, context):
        for metadata_type in self.get_metadata_types():
            try:
                context['workflow_instance'].document.metadata.get(
                    metadata_type=metadata_type
                ).delete()
            except DocumentMetadata.DoesNotExist:
                """This document does not have the metadata type added."""
            except ValidationError as exception:
                raise WorkflowStateActionError(
                    _(
                        message='Unable to remove metadata type "%(metadata_type)s" '
                        'from document: %(document)s. Exception: '
                        '%(exception)s'
                    ) % {
                        'document': context['workflow_instance'].document,
                        'exception': exception,
                        'metadata_type': metadata_type
                    }
                ) from exception
