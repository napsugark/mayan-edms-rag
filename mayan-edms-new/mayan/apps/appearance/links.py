from django.utils.translation import gettext_lazy as _

from mayan.apps.navigation.links import Link

from .icons import icon_ajax_refresh

link_ajax_refresh = Link(
    icon=icon_ajax_refresh, html_extra_classes='appearance-link-ajax-refresh',
    title=_(message='Reload the current content')
)
