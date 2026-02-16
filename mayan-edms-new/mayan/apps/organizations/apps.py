import logging

from django.utils.translation import gettext_lazy as _

from mayan.apps.app_manager.apps import MayanAppConfig

from .patches import patch_HttpRequest

logger = logging.getLogger(name=__name__)


class OrganizationsApp(MayanAppConfig):
    app_namespace = 'organizations'
    app_url = 'organizations'
    has_tests = True
    name = 'mayan.apps.organizations'
    verbose_name = _(message='Organizations')

    def ready(self):
        super().ready()

        patch_HttpRequest()
