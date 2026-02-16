import logging

from mayan.apps.acls.models import AccessControlList
from mayan.apps.navigation.links import Link

from ..exceptions import SourceActionExceptionUnknown
from ..icons import icon_upload_view_link
from ..links import factory_conditional_active_by_source
from ..menus import menu_sources
from ..models import Source

logger = logging.getLogger(name=__name__)


class SourceActionViewMixin:
    object_permission = None
    view_source_action = None

    def dispatch(self, request, *args, **kwargs):
        self.queryset_source_valid = self.get_queryset_source_valid()

        return super().dispatch(request=request, *args, **kwargs)

    def get_queryset_source_valid(self):
        def generator():
            view_source_action = self.get_view_source_action()

            queryset = Source.objects.filter(enabled=True)

            for source in queryset:
                try:
                    action = source.get_action(name=view_source_action)
                    if action.has_interface(interface_name='View'):
                        yield source.pk
                except SourceActionExceptionUnknown:
                    """
                    Source does not support the action, don't add it to the
                    id list of sources.
                    """

        queryset = Source.objects.filter(
            pk__in=generator()
        )

        if self.object_permission:
            queryset = AccessControlList.objects.restrict_queryset(
                permission=self.object_permission, queryset=queryset,
                user=self.request.user
            )

        return queryset

    def get_source_link_queryset(self):
        return self.source_link_queryset


class SourceLinkNavigationViewMixin:
    source_link_action = None
    source_link_permission = None
    source_link_queryset = None

    @staticmethod
    def get_source_link_for(source, view_name, view_kwargs=None):
        kwargs = {
            'source_id': str(source.pk)
        }

        if view_kwargs:
            kwargs.update(**view_kwargs)

        return Link(
            conditional_active=factory_conditional_active_by_source(
                source=source
            ), icon=icon_upload_view_link, keep_query=True,
            kwargs=kwargs, remove_from_query=['page'],
            text=source.label, view=view_name
        )

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        source_link_list = self.get_source_link_list()
        menu_sources.bound_links[
            'sources:document_upload'
        ] = source_link_list
        menu_sources.bound_links[
            'sources:document_file_upload'
        ] = source_link_list
        menu_sources.bound_links[
            'sources:source_action'
        ] = source_link_list

        return context

    def get_source_link_action(self):
        return self.source_link_action

    def get_source_link_list(self):
        queryset_sources = AccessControlList.objects.restrict_queryset(
            permission=self.get_source_link_permission(),
            queryset=self.get_source_link_queryset(),
            user=self.request.user
        )

        source_link_action = self.get_source_link_action()
        for source in queryset_sources:
            try:
                if source.get_action(name=source_link_action):
                    yield SourceLinkNavigationViewMixin.get_source_link_for(
                        source=source,
                        view_kwargs=self.get_source_link_view_kwargs(),
                        view_name=self.get_source_link_view_name()
                    )
            except SourceActionExceptionUnknown:
                """
                Source does not support the action. Ignore attempting
                to add the links.
                """

    def get_source_link_permission(self):
        return self.source_link_permission

    def get_source_link_queryset(self):
        return self.source_link_queryset

    def get_source_link_view_kwargs(self):
        return {}

    def get_source_link_view_name(self):
        return self.source_link_view_name
