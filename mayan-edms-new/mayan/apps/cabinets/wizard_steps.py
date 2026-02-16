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

from .forms import CabinetListForm
from .models import Cabinet
from .permissions import permission_cabinet_add_document


class DocumentCreateWizardStepCabinets(DocumentCreateWizardStep):
    form_class = CabinetListForm
    label = _(message='Select cabinets')
    name = 'cabinet_selection'
    number = 3

    @staticmethod
    def get_cabinets_queryset_for_document(document, user=None):
        queryset_documents = Document.valid.all()

        if user:
            queryset_documents = AccessControlList.objects.restrict_queryset(
                permission=permission_cabinet_add_document,
                queryset=queryset_documents, user=user
            )

        try:
            queryset_documents.get(pk=document.pk)
        except Document.DoesNotExist:
            queryset_cabinets = Cabinet.objects.none()
        else:
            queryset_cabinets = Cabinet.objects.all()

            if user:
                queryset_cabinets = AccessControlList.objects.restrict_queryset(
                    permission=permission_cabinet_add_document,
                    queryset=queryset_cabinets, user=user
                )

        return queryset_cabinets

    @staticmethod
    def get_cabinets_queryset_for_document_type(document_type, user):
        queryset_document_types = DocumentType.objects.all()

        queryset_document_types_restricted = AccessControlList.objects.restrict_queryset(
            permission=permission_cabinet_add_document,
            queryset=queryset_document_types, user=user
        )

        try:
            queryset_document_types_restricted.get(pk=document_type.pk)
        except DocumentType.DoesNotExist:
            queryset_cabinets = Cabinet.objects.none()
        else:
            queryset_cabinets = Cabinet.objects.all()

        return queryset_cabinets

    @classmethod
    def condition(cls, wizard):
        return Cabinet.objects.exists()

    @classmethod
    def get_form_kwargs(self, wizard):
        step_data = wizard.get_cleaned_data_for_step(
            step=DocumentCreateWizardStepDocumentType.name
        )
        user = wizard.request.user

        kwargs = {
            'help_text': _(
                message='Cabinets to which the document will be added.'
            ),
            'permission': permission_cabinet_add_document,
            'queryset': Cabinet.objects.none(),
            'user': user
        }

        if step_data:
            document_type = step_data['document_type']

            queryset_cabinets = DocumentCreateWizardStepCabinets.get_cabinets_queryset_for_document_type(
                document_type=document_type, user=user
            )

            kwargs['queryset'] = queryset_cabinets

        return kwargs

    @classmethod
    def done(cls, wizard):
        result = {}
        cleaned_data = wizard.get_cleaned_data_for_step(step=cls.name)
        if cleaned_data:
            result['cabinets'] = [
                str(cabinet.pk) for cabinet in cleaned_data['cabinets']
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

        queryset_cabinets = DocumentCreateWizardStepCabinets.get_cabinets_queryset_for_document(
            document=document, user=user
        )

        cabinet_id_list = URL(
            query_string=query_string
        ).args.getlist('cabinets')

        for cabinet in queryset_cabinets.filter(pk__in=cabinet_id_list):
            if user:
                cabinet.document_add(document=document, user=user)
            else:
                cabinet._document_add(document=document)


DocumentCreateWizardStep.register(step=DocumentCreateWizardStepCabinets)
