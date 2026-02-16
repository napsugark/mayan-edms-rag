from django.utils.translation import gettext_lazy as _

from mayan.apps.views.generics import SingleObjectListView

from .apps import MayanAppConfig
from .icons import icon_app_list


class AppListView(SingleObjectListView):
    view_icon = icon_app_list

    def get_extra_context(self):
        return {
            'hide_link': True,
            'hide_object': True,
            'subtitle': _(message='Detected and active Mayan EDMS apps.'),
            'title': _(message='System apps')
        }

    def get_source_queryset(self):
        return MayanAppConfig.get_all()
