from django.utils.translation import gettext_lazy as _

from mayan.apps.smart_settings.settings import setting_cluster

from .literals import (
    DEFAULT_REST_API_DISABLE_LINKS, DEFAULT_REST_API_MAXIMUM_PAGE_SIZE,
    DEFAULT_REST_API_PAGE_SIZE
)

setting_namespace = setting_cluster.do_namespace_add(
    label=_(message='REST API'), name='rest_api', version='0001'
)

setting_disable_links = setting_namespace.do_setting_add(
    choices=('false', 'true'), default=DEFAULT_REST_API_DISABLE_LINKS,
    global_name='REST_API_DISABLE_LINKS', help_text=_(
        message='Disable the REST API links in the tools menu.'
    )
)
setting_maximum_page_size = setting_namespace.do_setting_add(
    default=DEFAULT_REST_API_MAXIMUM_PAGE_SIZE,
    global_name='REST_API_MAXIMUM_PAGE_SIZE', help_text=_(
        message='The maximum page size that can be requested.'
    )
)
setting_page_size = setting_namespace.do_setting_add(
    default=DEFAULT_REST_API_PAGE_SIZE,
    global_name='REST_API_PAGE_SIZE', help_text=_(
        message='The default page size if none is specified.'
    )
)
