from django.utils.translation import gettext_lazy as _

from mayan.apps.dashboards.classes import Dashboard

dashboard_user = Dashboard(
    name='user', label=_(message='User dashboard')
)
