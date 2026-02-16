from rest_framework.generics import get_object_or_404

from mayan.apps.acls.models import AccessControlList

from ..models.document_models import Document
from ..models.document_type_models import DocumentType


class ParentObjectDocumentAPIViewMixin:
    def get_document(self, permission=None):
        queryset = self.get_document_queryset()

        if not permission:
            mayan_external_object_permission_map = getattr(
                self, 'mayan_external_object_permission_map', {}
            )

            permission = mayan_external_object_permission_map.get(
                self.request.method, None
            )

        if permission:
            queryset = AccessControlList.objects.restrict_queryset(
                permission=permission, queryset=queryset,
                user=self.request.user
            )

        return get_object_or_404(
            queryset=queryset, pk=self.kwargs['document_id']
        )

    def get_document_queryset(self):
        return Document.valid.all()


class ParentObjectDocumentFileAPIViewMixin(ParentObjectDocumentAPIViewMixin):
    def get_document_file(self, permission=None):
        queryset = self.get_document_file_queryset()

        if permission:
            queryset = AccessControlList.objects.restrict_queryset(
                permission=permission, queryset=queryset,
                user=self.request.user
            )

        return get_object_or_404(
            queryset=queryset, pk=self.kwargs['document_file_id']
        )

    def get_document_file_queryset(self):
        document = self.get_document()
        return document.files.all()


class ParentObjectDocumentFilePageAPIViewMixin(
    ParentObjectDocumentFileAPIViewMixin
):
    def get_document_file_page(self, permission=None):
        queryset = self.get_document_file_page_queryset()

        if permission:
            queryset = AccessControlList.objects.restrict_queryset(
                permission=permission, queryset=queryset,
                user=self.request.user
            )

        return get_object_or_404(
            queryset=queryset, pk=self.kwargs['document_file_page_id']
        )

    def get_document_file_page_queryset(self):
        document_file = self.get_document_file()
        return document_file.pages.all()


class ParentObjectDocumentTypeAPIViewMixin:
    def get_document_type(self, permission=None):
        queryset = DocumentType.objects.all()

        if not permission:
            mayan_external_object_permission_map = getattr(
                self, 'mayan_external_object_permission_map', {}
            )

            permission = mayan_external_object_permission_map.get(
                self.request.method, None
            )

        if permission:
            queryset = AccessControlList.objects.restrict_queryset(
                permission=permission, queryset=queryset,
                user=self.request.user
            )

        return get_object_or_404(
            queryset=queryset, pk=self.kwargs['document_type_id']
        )


class ParentObjectDocumentVersionAPIViewMixin(ParentObjectDocumentAPIViewMixin):
    def get_document_version(self, permission=None):
        queryset = self.get_document_version_queryset()

        if permission:
            queryset = AccessControlList.objects.restrict_queryset(
                permission=permission, queryset=queryset,
                user=self.request.user
            )

        return get_object_or_404(
            queryset=queryset, pk=self.kwargs['document_version_id']
        )

    def get_document_version_queryset(self):
        document = self.get_document()
        return document.versions.all()


class ParentObjectDocumentVersionPageAPIViewMixin(
    ParentObjectDocumentVersionAPIViewMixin
):
    def get_document_version_page(self, permission=None):
        queryset = self.get_document_version_page_queryset()

        if permission:
            queryset = AccessControlList.objects.restrict_queryset(
                permission=permission, queryset=queryset,
                user=self.request.user
            )

        return get_object_or_404(
            queryset=queryset, pk=self.kwargs['document_version_page_id']
        )

    def get_document_version_page_queryset(self):
        document_version = self.get_document_version()
        return document_version.pages.all()
