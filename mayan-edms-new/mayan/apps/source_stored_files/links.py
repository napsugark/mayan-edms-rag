from django.utils.translation import gettext_lazy as _

from mayan.apps.documents.permissions import permission_document_create
from mayan.apps.navigation.links import Link

from .icons import icon_storage_file_delete, icon_storage_file_select

link_storage_file_delete = Link(
    icon=icon_storage_file_delete, kwargs={
        'source_id': 'source.pk',
        'action_name': '"file_delete"'
    }, permission=permission_document_create, query={
        'document_id': 'document.pk',
        'document_type_id': 'document_type.pk',
        'source_id': 'source.pk',
        'encoded_filename': 'object.encoded_filename'
    }, tags='dangerous', text=_(message='Delete'),
    view='sources:source_action'
)
link_source_file_select = Link(
    html_data={'encoded_filename': 'object.encoded_filename'},
    html_extra_classes='source_stored_files-stored_file-select',
    icon=icon_storage_file_select, text=_(message='Select'), url=''
)
