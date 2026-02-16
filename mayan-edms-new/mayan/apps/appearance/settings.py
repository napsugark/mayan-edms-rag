from django.utils.translation import gettext_lazy as _

from mayan.apps.smart_settings.settings import setting_cluster

from .literals import (
    DEFAULT_APPEARANCE_ELIDED_PAGER_ON_EACH_SIDE,
    DEFAULT_APPEARANCE_ELIDED_PAGER_ON_ENDS,
    DEFAULT_APPEARANCE_PAGINATION_DROPDOWN_RANGE,
    DEFAULT_APPEARANCE_PAGINATION_DROPDOWN_ENABLE,
    DEFAULT_APPEARANCE_PAGINATION_INPUT_ENABLE,
    DEFAULT_AJAX_REDIRECTION_CODE, DEFAULT_MAXIMUM_TITLE_LENGTH,
    DEFAULT_MENU_POLLING_INTERVAL, DEFAULT_MESSAGE_POSITION,
    DEFAULT_THROTTLING_MAXIMUM_REQUESTS, DEFAULT_THROTTLING_TIMEOUT
)

setting_namespace = setting_cluster.do_namespace_add(
    label=_(message='Appearance'), name='appearance'
)

setting_ajax_redirection_code = setting_namespace.do_setting_add(
    default=DEFAULT_AJAX_REDIRECTION_CODE,
    global_name='APPEARANCE_AJAX_REDIRECTION_CODE', help_text=_(
        message='Custom HTTP response code for AJAX redirections.'
    )
)
setting_pagination_input_threshold = setting_namespace.do_setting_add(
    default=DEFAULT_APPEARANCE_ELIDED_PAGER_ON_EACH_SIDE,
    global_name='APPEARANCE_ELIDED_PAGER_ON_EACH_SIDE', help_text=_(
        message='Number of pages to show on each side of the current '
        'page in the elided pager.'
    )
)
setting_pagination_input_threshold = setting_namespace.do_setting_add(
    default=DEFAULT_APPEARANCE_ELIDED_PAGER_ON_ENDS,
    global_name='APPEARANCE_ELIDED_PAGER_ON_ENDS', help_text=_(
        message='Number of pages to show at both ends of the elided pager.'
    )
)
setting_max_title_length = setting_namespace.do_setting_add(
    default=DEFAULT_MAXIMUM_TITLE_LENGTH,
    global_name='APPEARANCE_MAXIMUM_TITLE_LENGTH', help_text=_(
        message='Maximum number of characters that will be displayed as the view '
        'title.'
    )
)
setting_message_position = setting_namespace.do_setting_add(
    choices=(
        'top-left', 'top-center', 'top-right', 'bottom-left',
        'bottom-center', 'bottom-right',
    ), default=DEFAULT_MESSAGE_POSITION,
    global_name='APPEARANCE_MESSAGE_POSITION', help_text=_(
        message='Position where the system messages will be displayed.'
    )
)
setting_menu_polling_interval = setting_namespace.do_setting_add(
    default=DEFAULT_MENU_POLLING_INTERVAL,
    global_name='APPEARANCE_MENU_POLLING_INTERVAL', help_text=_(
        message='Delay in milliseconds after which the menus will be checked for '
        'updates.'
    )
)
setting_pagination_dropdown_range = setting_namespace.do_setting_add(
    default=DEFAULT_APPEARANCE_PAGINATION_DROPDOWN_RANGE,
    global_name='APPEARANCE_PAGINATION_DROPDOWN_RANGE', help_text=_(
        message='Total pages to show in the page selection dropdown.'
    )
)
setting_pagination_dropdown_threshold = setting_namespace.do_setting_add(
    default=DEFAULT_APPEARANCE_PAGINATION_DROPDOWN_ENABLE,
    global_name='APPEARANCE_PAGINATION_DROPDOWN_ENABLE', help_text=_(
        message='Enable page selection dropdown.'
    )
)
setting_pagination_input_threshold = setting_namespace.do_setting_add(
    default=DEFAULT_APPEARANCE_PAGINATION_INPUT_ENABLE,
    global_name='APPEARANCE_PAGINATION_INPUT_ENABLE', help_text=_(
        message='Enable the page selection input field.'
    )
)
setting_throttling_maximum_requests = setting_namespace.do_setting_add(
    default=DEFAULT_THROTTLING_MAXIMUM_REQUESTS,
    global_name='APPEARANCE_THROTTLING_MAXIMUM_REQUESTS', help_text=_(
        message='Maximum number of requests that can be made before throttling '
        'is enabled.'
    )
)
setting_throttling_timeout = setting_namespace.do_setting_add(
    default=DEFAULT_THROTTLING_TIMEOUT,
    global_name='APPEARANCE_THROTTLING_TIMEOUT', help_text=_(
        message='Time in milliseconds after which a throttled request will '
        'clear allowing an additional request to be performed.'
    )
)
