from django.urls import re_path

from .api_views.index_instance_api_views import (
    APIDocumentIndexInstanceNodeListView, APIIndexInstanceDetailView,
    APIIndexInstanceListView, APIIndexInstanceNodeChildrenNodeListView,
    APIIndexInstanceNodeDetailView, APIIndexInstanceNodeDocumentListView,
    APIIndexInstanceNodeListView
)
from .api_views.index_template_api_views import (
    APIIndexTemplateDetailView, APIIndexTemplateDocumentTypeAddView,
    APIIndexTemplateDocumentTypeListView,
    APIIndexTemplateDocumentTypeRemoveView, APIIndexTemplateListView,
    APIIndexTemplateNodeDetailView, APIIndexTemplateNodeListView,
    APIIndexTemplateRebuildView, APIIndexTemplateResetView
)
from .views.index_instance_views import (
    DocumentIndexInstanceNodeListView, IndexInstanceListView,
    IndexInstanceNodeView
)
from .views.index_template_views import (
    DocumentTypeIndexTemplateAddRemoveView, IndexTemplateAllRebuildView,
    IndexTemplateCreateView, IndexTemplateDeleteView,
    IndexTemplateDocumentTypeAddRemoveView, IndexTemplateEditView,
    IndexTemplateEventTriggerListView, IndexTemplateListView,
    IndexTemplateNodeCreateView, IndexTemplateNodeDeleteView,
    IndexTemplateNodeEditView, IndexTemplateNodeListView,
    IndexTemplateRebuildView, IndexTemplateResetView
)

urlpatterns_templates = [
    re_path(
        route=r'^document_types/(?P<document_type_id>\d+)/index_templates/$',
        name='document_type_index_templates',
        view=DocumentTypeIndexTemplateAddRemoveView.as_view()
    ),
    re_path(
        route=r'^templates/$', name='index_template_list',
        view=IndexTemplateListView.as_view()
    ),
    re_path(
        route=r'^templates/create/$', name='index_template_create',
        view=IndexTemplateCreateView.as_view()
    ),
    re_path(
        route=r'^templates/(?P<index_template_id>\d+)/delete/$',
        name='index_template_delete', view=IndexTemplateDeleteView.as_view()
    ),
    re_path(
        route=r'^templates/(?P<index_template_id>\d+)/document_types/$',
        name='index_template_document_types',
        view=IndexTemplateDocumentTypeAddRemoveView.as_view()
    ),
    re_path(
        route=r'^templates/(?P<index_template_id>\d+)/edit/$',
        name='index_template_edit', view=IndexTemplateEditView.as_view()
    ),
    re_path(
        route=r'^templates/(?P<index_template_id>\d+)/event_triggers/$',
        name='index_template_event_triggers',
        view=IndexTemplateEventTriggerListView.as_view()
    ),
    re_path(
        route=r'^templates/(?P<index_template_id>\d+)/nodes/$',
        name='index_template_view',
        view=IndexTemplateNodeListView.as_view()
    ),
    re_path(
        route=r'^templates/(?P<index_template_id>\d+)/rebuild/$',
        name='index_template_rebuild',
        view=IndexTemplateRebuildView.as_view()
    ),
    re_path(
        route=r'^templates/nodes/(?P<index_template_node_id>\d+)/children/create/$',
        name='template_node_create',
        view=IndexTemplateNodeCreateView.as_view()
    ),
    re_path(
        route=r'^templates/nodes/(?P<index_template_node_id>\d+)/delete/$',
        name='template_node_delete',
        view=IndexTemplateNodeDeleteView.as_view()
    ),
    re_path(
        route=r'^templates/nodes/(?P<index_template_node_id>\d+)/edit/$',
        name='template_node_edit', view=IndexTemplateNodeEditView.as_view()
    )
]

urlpatterns_instances = [
    re_path(
        route=r'^instances/$', name='index_list',
        view=IndexInstanceListView.as_view()
    ),
    re_path(
        route=r'^instances/nodes/(?P<index_instance_node_id>\d+)/$',
        name='index_instance_node_view', view=IndexInstanceNodeView.as_view()
    ),
    re_path(
        route=r'^documents/(?P<document_id>\d+)/instances/$',
        name='document_index_list',
        view=DocumentIndexInstanceNodeListView.as_view()
    )
]

urlpatterns_tools = [
    re_path(
        route=r'^instances/rebuild/$', name='rebuild_index_instances',
        view=IndexTemplateAllRebuildView.as_view()
    ),
    re_path(
        route=r'^instances/reset/$', name='index_instances_reset',
        view=IndexTemplateResetView.as_view()
    )
]

urlpatterns = []
urlpatterns.extend(urlpatterns_templates)
urlpatterns.extend(urlpatterns_instances)
urlpatterns.extend(urlpatterns_tools)

api_urls_document_indexes = [
    re_path(
        route=r'^documents/(?P<document_id>[0-9]+)/indexes/$',
        name='document-index-list',
        view=APIDocumentIndexInstanceNodeListView.as_view()
    )
]

api_urls_index_instances = [
    re_path(
        route=r'^index_instances/$', name='indexinstance-list',
        view=APIIndexInstanceListView.as_view()
    ),
    re_path(
        route=r'^index_instances/(?P<index_instance_id>[0-9]+)/$',
        name='indexinstance-detail',
        view=APIIndexInstanceDetailView.as_view()
    ),
    re_path(
        route=r'^index_instances/(?P<index_instance_id>[0-9]+)/nodes/$',
        name='indexinstancenode-list',
        view=APIIndexInstanceNodeListView.as_view()
    ),
    re_path(
        route=r'^index_instances/(?P<index_instance_id>[0-9]+)/nodes/(?P<index_instance_node_id>[0-9]+)/$',
        name='indexinstancenode-detail',
        view=APIIndexInstanceNodeDetailView.as_view()
    ),
    re_path(
        route=r'^index_instances/(?P<index_instance_id>[0-9]+)/nodes/(?P<index_instance_node_id>[0-9]+)/nodes/$',
        name='indexinstancenode-children-list',
        view=APIIndexInstanceNodeChildrenNodeListView.as_view()
    ),
    re_path(
        route=r'^index_instances/(?P<index_instance_id>[0-9]+)/nodes/(?P<index_instance_node_id>[0-9]+)/documents/$',
        name='indexinstancenode-document-list',
        view=APIIndexInstanceNodeDocumentListView.as_view()
    )
]

api_urls_index_templates = [
    re_path(
        route=r'^index_templates/$', name='indextemplate-list',
        view=APIIndexTemplateListView.as_view()
    ),
    re_path(
        route=r'^index_templates/(?P<index_template_id>[0-9]+)/$',
        name='indextemplate-detail',
        view=APIIndexTemplateDetailView.as_view()
    ),
    re_path(
        route=r'^index_templates/(?P<index_template_id>[0-9]+)/document_types/$',
        name='indextemplate-documenttype-list',
        view=APIIndexTemplateDocumentTypeListView.as_view()
    ),
    re_path(
        route=r'^index_templates/(?P<index_template_id>[0-9]+)/document_types/add/$',
        name='indextemplate-documenttype-add',
        view=APIIndexTemplateDocumentTypeAddView.as_view()
    ),
    re_path(
        route=r'^index_templates/(?P<index_template_id>[0-9]+)/document_types/remove/$',
        name='indextemplate-documenttype-remove',
        view=APIIndexTemplateDocumentTypeRemoveView.as_view()
    ),
    re_path(
        route=r'^index_templates/(?P<index_template_id>[0-9]+)/rebuild/$',
        name='indextemplate-rebuild',
        view=APIIndexTemplateRebuildView.as_view()
    ),
    re_path(
        route=r'^index_templates/(?P<index_template_id>[0-9]+)/reset/$',
        name='indextemplate-reset',
        view=APIIndexTemplateResetView.as_view()
    ),
    re_path(
        route=r'^index_templates/(?P<index_template_id>[0-9]+)/nodes/$',
        name='indextemplatenode-list',
        view=APIIndexTemplateNodeListView.as_view()
    ),
    re_path(
        route=r'^index_templates/(?P<index_template_id>[0-9]+)/nodes/(?P<index_template_node_id>[0-9]+)/$',
        name='indextemplatenode-detail',
        view=APIIndexTemplateNodeDetailView.as_view()
    )
]

api_urls = []
api_urls.extend(api_urls_document_indexes)
api_urls.extend(api_urls_index_instances)
api_urls.extend(api_urls_index_templates)
