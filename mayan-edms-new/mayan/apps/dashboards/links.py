from django.utils.translation import gettext_lazy as _

from mayan.apps.navigation.links import Link

from .icons import icon_dashboard_detail, icon_dashboard_list

link_dashboard_detail = Link(
    args='object.name', icon=icon_dashboard_detail,
    text=_(message='Details'), view='dashboards:dashboard_detail'
)
link_dashboard_list = Link(
    icon=icon_dashboard_list, text=_(message='Dashboards'),
    view='dashboards:dashboard_list'
)
