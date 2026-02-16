from django.apps import apps
from django.contrib.auth import get_user_model
from django.utils.translation import gettext_lazy as _

from mayan.apps.acls.models import AccessControlList
from mayan.apps.documents.models.document_models import Document
from mayan.apps.documents.models.document_type_models import DocumentType
from mayan.apps.sources.classes import DocumentCreateWizardStep
from mayan.apps.sources.wizard_steps import (
    DocumentCreateWizardStepDocumentType
)
from mayan.apps.views.http import URL

from .forms import TagMultipleSelectionForm
from .models import Tag
from .permissions import permission_tag_attach


class DocumentCreateWizardStepTags(DocumentCreateWizardStep):
    form_class = TagMultipleSelectionForm
    label = _(message='Select tags')
    name = 'tag_selection'
    number = 2

    @staticmethod
    def get_tags_queryset_for_document(document, user=None):
        queryset_documents = Document.valid.all()

        if user:
            queryset_documents = AccessControlList.objects.restrict_queryset(
                permission=permission_tag_attach,
                queryset=queryset_documents, user=user
            )

        try:
            queryset_documents.get(pk=document.pk)
        except Document.DoesNotExist:
            queryset_tags = Tag.objects.none()
        else:
            queryset_tags = Tag.objects.all()

            if user:
                queryset_tags = AccessControlList.objects.restrict_queryset(
                    permission=permission_tag_attach, queryset=queryset_tags,
                    user=user
                )

        return queryset_tags

    @staticmethod
    def get_tags_queryset_for_document_type(document_type, user):
        queryset_document_types = DocumentType.objects.all()

        queryset_document_types_restricted = AccessControlList.objects.restrict_queryset(
            permission=permission_tag_attach,
            queryset=queryset_document_types, user=user
        )

        try:
            queryset_document_types_restricted.get(pk=document_type.pk)
        except DocumentType.DoesNotExist:
            queryset_tags = Tag.objects.none()
        else:
            queryset_tags = Tag.objects.all()

        return queryset_tags

    @classmethod
    def condition(cls, wizard):
        Tag = apps.get_model(app_label='tags', model_name='Tag')

        return Tag.objects.exists()

    @classmethod
    def get_form_kwargs(self, wizard):
        step_data = wizard.get_cleaned_data_for_step(
            step=DocumentCreateWizardStepDocumentType.name
        )
        user = wizard.request.user

        kwargs = {
            'help_text': _(message='Tags to be attached.'),
            'permission': permission_tag_attach,
            'queryset': Tag.objects.none(),
            'user': user
        }

        if step_data:
            document_type = step_data['document_type']

            queryset_tags = DocumentCreateWizardStepTags.get_tags_queryset_for_document_type(
                document_type=document_type, user=user
            )

            kwargs['queryset'] = queryset_tags

        return kwargs

    @classmethod
    def done(cls, wizard):
        result = {}
        cleaned_data = wizard.get_cleaned_data_for_step(cls.name)
        if cleaned_data:
            result['tags'] = [
                str(tag.pk) for tag in cleaned_data['tags']
            ]

        return result

    @classmethod
    def step_post_upload_process(
        cls, document, query_string, source_id, user_id
    ):
        User = get_user_model()

        try:
            user = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            user = None

        queryset_tags = DocumentCreateWizardStepTags.get_tags_queryset_for_document(
            document=document, user=user
        )

        tag_id_list = URL(query_string=query_string).args.getlist('tags')

        for tag in queryset_tags.filter(pk__in=tag_id_list):
            if user:
                tag.attach_to(document=document, user=user)
            else:
                tag._attach_to(document=document)


DocumentCreateWizardStep.register(step=DocumentCreateWizardStepTags)
