from django.urls import re_path

from .api_views import (
    APIMailObjectAttachmentView, APIMailObjectLinkView,
    APIMailingProfileDetailView, APIMailingProfileListView
)
from .views.document_file_views import (
    MailDocumentFileAttachmentView, MailDocumentFileLinkView
)
from .views.document_version_views import (
    MailDocumentVersionAttachmentView, MailDocumentVersionLinkView
)
from .views.document_views import MailDocumentLinkView
from .views.mailing_profile_views import (
    MailingProfileBackendSelectionView, MailingProfileCreateView,
    MailingProfileDeleteView, MailingProfileEditView, MailingProfileListView,
    MailingProfileTestView
)

urlpatterns_document = [
    re_path(
        route=r'^documents/(?P<document_id>\d+)/send/link/$',
        name='send_document_link_single',
        view=MailDocumentLinkView.as_view()
    ),
    re_path(
        route=r'^documents/multiple/send/link/$',
        name='send_document_link_multiple',
        view=MailDocumentLinkView.as_view()
    )
]

urlpatterns_document_file = [
    re_path(
        route=r'^documents/files/(?P<document_file_id>\d+)/send/attachment/$',
        name='send_document_file_attachment_single',
        view=MailDocumentFileAttachmentView.as_view()
    ),
    re_path(
        route=r'^documents/files/multiple/send/attachment/$',
        name='send_document_file_attachment_multiple',
        view=MailDocumentFileAttachmentView.as_view()
    ),
    re_path(
        route=r'^documents/files/(?P<document_file_id>\d+)/send/link/$',
        name='send_document_file_link_single',
        view=MailDocumentFileLinkView.as_view()
    ),
    re_path(
        route=r'^documents/files/multiple/send/link/$',
        name='send_document_file_link_multiple',
        view=MailDocumentFileLinkView.as_view()
    )
]

urlpatterns_document_version = [
    re_path(
        route=r'^documents/versions/(?P<document_version_id>\d+)/send/attachment/$',
        name='send_document_version_attachment_single',
        view=MailDocumentVersionAttachmentView.as_view()
    ),
    re_path(
        route=r'^documents/versions/multiple/send/attachment/$',
        name='send_document_version_attachment_multiple',
        view=MailDocumentVersionAttachmentView.as_view()
    ),
    re_path(
        route=r'^documents/versions/(?P<document_version_id>\d+)/send/link/$',
        name='send_document_version_link_single',
        view=MailDocumentVersionLinkView.as_view()
    ),
    re_path(
        route=r'^documents/versions/multiple/send/link/$',
        name='send_document_version_link_multiple',
        view=MailDocumentVersionLinkView.as_view()
    )
]

urlpatterns_mailing_profiles = [
    re_path(
        route=r'^mailing_profiles/backend/selection/$',
        name='mailing_profile_backend_selection',
        view=MailingProfileBackendSelectionView.as_view()
    ),
    re_path(
        route=r'^mailing_profiles/(?P<backend_path>[a-zA-Z0-9_.]+)/create/$',
        name='mailing_profile_create', view=MailingProfileCreateView.as_view()
    ),
    re_path(
        route=r'^mailing_profiles/(?P<mailing_profile_id>\d+)/delete/$',
        name='mailing_profile_delete', view=MailingProfileDeleteView.as_view()
    ),
    re_path(
        route=r'^mailing_profiles/(?P<mailing_profile_id>\d+)/edit/$',
        name='mailing_profile_edit', view=MailingProfileEditView.as_view()
    ),
    re_path(
        route=r'^mailing_profiles/(?P<mailing_profile_id>\d+)/test/$',
        name='mailing_profile_test', view=MailingProfileTestView.as_view()
    ),
    re_path(
        route=r'^mailing_profiles/$', name='mailing_profile_list',
        view=MailingProfileListView.as_view()
    )
]

urlpatterns = []
urlpatterns.extend(urlpatterns_document)
urlpatterns.extend(urlpatterns_document_file)
urlpatterns.extend(urlpatterns_document_version)
urlpatterns.extend(urlpatterns_mailing_profiles)

api_urls_mailing_profiles = [
    re_path(
        route=r'^mailing_profiles/$', name='mailing_profile-list',
        view=APIMailingProfileListView.as_view()
    ),
    re_path(
        route=r'^mailing_profiles/(?P<mailing_profile_id>[0-9]+)/$',
        name='mailing_profile-detail',
        view=APIMailingProfileDetailView.as_view()
    )
]

api_urls_mailing_profile_actions = [
    re_path(
        route=r'^objects/(?P<app_label>[-\w]+)/(?P<model_name>[-\w]+)/(?P<object_id>\d+)/mailing/attachment/$',
        name='object-mailing-action-attachment',
        view=APIMailObjectAttachmentView.as_view()
    ),
    re_path(
        route=r'^objects/(?P<app_label>[-\w]+)/(?P<model_name>[-\w]+)/(?P<object_id>\d+)/mailing/link/$',
        name='object-mailing-action-link', view=APIMailObjectLinkView.as_view()
    )
]

api_urls = []
api_urls.extend(api_urls_mailing_profiles)
api_urls.extend(api_urls_mailing_profile_actions)
