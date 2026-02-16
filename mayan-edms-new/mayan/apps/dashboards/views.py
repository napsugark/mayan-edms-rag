from django.utils.translation import gettext_lazy as _

from mayan.apps.views.generics import SimpleView, SingleObjectListView

from .classes import Dashboard
from .icons import icon_dashboard_detail, icon_dashboard_list


class DashboardListView(SingleObjectListView):
    extra_context = {
        'hide_object': True,
        'title': _(message='Dashboards')
    }
    view_icon = icon_dashboard_list

    def get_source_queryset(self):
        return Dashboard.get_all()


class DashboardDetailView(SimpleView):
    template_name = 'appearance/content_container.html'
    view_icon = icon_dashboard_detail

    def get_extra_context(self):
        dashboard = Dashboard.get(
            name=self.kwargs['dashboard_name']
        )

        return {
            'content': dashboard.render(request=self.request),
            'object': dashboard,
            'title': _(message='Dashboard detail')
        }
