from django.urls import re_path

from .api_views import (
    APICheckedoutDocumentListView, APICheckedoutDocumentView,
    APIDocumentCheckoutView
)
from .views import (
    DocumentCheckInView, DocumentCheckOutDetailView, DocumentCheckOutListView,
    DocumentCheckOutView
)

urlpatterns = [
    re_path(
        route=r'^documents/$', name='check_out_list',
        view=DocumentCheckOutListView.as_view()
    ),
    re_path(
        route=r'^documents/(?P<document_id>\d+)/check/in/$',
        name='check_in_document', view=DocumentCheckInView.as_view()
    ),
    re_path(
        route=r'^documents/multiple/check/in/$',
        name='check_in_document_multiple', view=DocumentCheckInView.as_view()
    ),
    re_path(
        route=r'^documents/(?P<document_id>\d+)/check/out/$',
        name='check_out_document', view=DocumentCheckOutView.as_view()
    ),
    re_path(
        route=r'^documents/multiple/check/out/$',
        name='check_out_document_multiple',
        view=DocumentCheckOutView.as_view()
    ),
    re_path(
        route=r'^documents/(?P<document_id>\d+)/checkout/info/$',
        name='check_out_info', view=DocumentCheckOutDetailView.as_view()
    )
]

api_urls = [
    re_path(
        route=r'^checkouts/$', name='checkout-document-list',
        view=APICheckedoutDocumentListView.as_view()
    ),
    re_path(
        route=r'^checkouts/(?P<checkout_id>[0-9]+)/checkout_info/$',
        name='checkedout-document-view',
        view=APICheckedoutDocumentView.as_view()
    ),
    re_path(
        route=r'^documents/(?P<document_id>[0-9]+)/checkout/$',
        name='document-checkout-view',
        view=APIDocumentCheckoutView.as_view()
    )
]
