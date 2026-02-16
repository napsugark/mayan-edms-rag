from django.urls import re_path

from .api_views import (
    APIObjectTemplateSandboxActionView, APITemplateDetailView, APITemplateListView
)

from .views import ObjectTemplateSandboxView

urlpatterns = [
    re_path(
        route=r'^objects/(?P<app_label>[-\w]+)/(?P<model_name>[-\w]+)/(?P<object_id>\d+)/sandbox/$',
        name='object_template_sandbox',
        view=ObjectTemplateSandboxView.as_view()
    )
]

api_urls = [
    re_path(
        route=r'^objects/(?P<app_label>[-\w]+)/(?P<model_name>[-\w]+)/(?P<object_id>\d+)/sandbox/$',
        view=APIObjectTemplateSandboxActionView.as_view(),
        name='object-template-sandbox'
    ),
    re_path(
        route=r'^templates/$', view=APITemplateListView.as_view(),
        name='template-list'
    ),
    re_path(
        route=r'^templates/(?P<name>[-\w]+)/$',
        view=APITemplateDetailView.as_view(), name='template-detail'
    )
]
