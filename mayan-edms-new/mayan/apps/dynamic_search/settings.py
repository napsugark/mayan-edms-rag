from django.utils.translation import gettext_lazy as _

from mayan.apps.smart_settings.settings import setting_cluster

from .literals import (
    DEFAULT_SEARCH_BACKEND, DEFAULT_SEARCH_BACKEND_ARGUMENTS,
    DEFAULT_SEARCH_DEFAULT_OPERATOR, DEFAULT_SEARCH_DISABLE_SIMPLE_SEARCH,
    DEFAULT_SEARCH_INDEXING_CHUNK_SIZE,
    DEFAULT_SEARCH_MATCH_ALL_DEFAULT_VALUE,
    DEFAULT_SEARCH_MODEL_FIELD_DISABLE, DEFAULT_SEARCH_QUERY_RESULTS_LIMIT,
    DEFAULT_SEARCH_QUERY_RESULTS_LIMIT_ERROR, DEFAULT_SEARCH_RESULTS_LIMIT,
    DEFAULT_SEARCH_SAVED_RESULTSET_RESULTS_LIMIT,
    DEFAULT_SEARCH_SAVED_RESULTSETS_PER_USER_LIMIT,
    DEFAULT_SEARCH_SAVED_RESULTSET_TIME_TO_LIVE,
    DEFAULT_SEARCH_SAVED_RESULTSET_TIME_TO_LIVE_INCREMENT,
    DEFAULT_SEARCH_STORE_RESULTS_DEFAULT_VALUE, SCOPE_OPERATOR_CHOICES
)

setting_namespace = setting_cluster.do_namespace_add(
    label=_(message='Search'), name='search'
)

setting_backend = setting_namespace.do_setting_add(
    default=DEFAULT_SEARCH_BACKEND, global_name='SEARCH_BACKEND', help_text=_(
        message='Full path to the backend to be used to handle the search.'
    )
)
setting_backend_arguments = setting_namespace.do_setting_add(
    default=DEFAULT_SEARCH_BACKEND_ARGUMENTS,
    global_name='SEARCH_BACKEND_ARGUMENTS', help_text=_(
        message='Arguments to pass to the search backend. For example values '
        'to change the behavior, host names, or authentication arguments.'
    )
)
setting_default_operator = setting_namespace.do_setting_add(
    choices=SCOPE_OPERATOR_CHOICES.keys(),
    global_name='SEARCH_DEFAULT_OPERATOR',
    default=DEFAULT_SEARCH_DEFAULT_OPERATOR, help_text=_(
        message='The search operator to use when none is specified.'
    )
)
setting_disable_simple_search = setting_namespace.do_setting_add(
    choices=('false', 'true'),
    default=DEFAULT_SEARCH_DISABLE_SIMPLE_SEARCH,
    global_name='SEARCH_DISABLE_SIMPLE_SEARCH', help_text=_(
        message='Disables the single term bar search leaving only the '
        'advanced search button.'
    )
)
setting_indexing_chunk_size = setting_namespace.do_setting_add(
    default=DEFAULT_SEARCH_INDEXING_CHUNK_SIZE,
    global_name='SEARCH_INDEXING_CHUNK_SIZE', help_text=_(
        message='Amount of objects to process when performing bulk indexing.'
    )
)
setting_match_all_default_value = setting_namespace.do_setting_add(
    global_name='SEARCH_MATCH_ALL_DEFAULT_VALUE',
    default=DEFAULT_SEARCH_MATCH_ALL_DEFAULT_VALUE, help_text=_(
        message='Sets the default state of the "Match all" checkbox.'
    )
)
setting_query_results_limit = setting_namespace.do_setting_add(
    default=DEFAULT_SEARCH_QUERY_RESULTS_LIMIT,
    global_name='SEARCH_QUERY_RESULTS_LIMIT', help_text=_(
        message='Maximum number of search results to fetch and display per '
        'search query unit.'
    )
)
setting_query_results_limit_error = setting_namespace.do_setting_add(
    default=DEFAULT_SEARCH_QUERY_RESULTS_LIMIT_ERROR,
    global_name='SEARCH_QUERY_RESULTS_LIMIT_ERROR', help_text=_(
        message='Raise an error when the number of search results '
        'exceed the value of `SEARCH_QUERY_RESULTS_LIMIT`.'
    )
)
setting_results_limit = setting_namespace.do_setting_add(
    default=DEFAULT_SEARCH_RESULTS_LIMIT, global_name='SEARCH_RESULTS_LIMIT',
    help_text=_(
        message='Maximum number of search results to fetch and display.'
    )
)
setting_saved_resultsets_per_user_limit = setting_namespace.do_setting_add(
    default=DEFAULT_SEARCH_SAVED_RESULTSETS_PER_USER_LIMIT,
    global_name='SEARCH_SAVED_RESULTSETS_PER_USER_LIMIT', help_text=_(
        message='Maximum number of saved resultsets to keep per user.'
    )
)
setting_saved_resultset_results_limit = setting_namespace.do_setting_add(
    default=DEFAULT_SEARCH_SAVED_RESULTSET_RESULTS_LIMIT,
    global_name='SEARCH_SAVED_RESULTSET_RESULTS_LIMIT', help_text=_(
        message='Maximum number of results to store per resultset.'
    )
)
setting_saved_resultset_time_to_live = setting_namespace.do_setting_add(
    default=DEFAULT_SEARCH_SAVED_RESULTSET_TIME_TO_LIVE,
    global_name='SEARCH_SAVED_RESULTSET_TIME_TO_LIVE', help_text=_(
        message='Time to keep the resultset in seconds.'
    )
)
setting_saved_resultset_time_to_live_increment = setting_namespace.do_setting_add(
    default=DEFAULT_SEARCH_SAVED_RESULTSET_TIME_TO_LIVE_INCREMENT,
    global_name='SEARCH_SAVED_RESULTSET_TIME_TO_LIVE_INCREMENT', help_text=_(
        message='Amount to increase the time to live on each access of the '
        'resultset.'
    )
)
setting_search_model_field_disable = setting_namespace.do_setting_add(
    global_name='SEARCH_MODEL_FIELD_DISABLE',
    default=DEFAULT_SEARCH_MODEL_FIELD_DISABLE, help_text=_(
        message='Specifies the fields from which search model are to be '
        'disabled. The format is a dictionary of lists. The search model '
        'name is the dictionary key and the value is a list of the full '
        'search field name. Disabled fields will neither be available for '
        'search nor be indexed by the search backend.'
    )
)
setting_store_results_default_value = setting_namespace.do_setting_add(
    global_name='SEARCH_STORE_RESULTS_DEFAULT_VALUE',
    default=DEFAULT_SEARCH_STORE_RESULTS_DEFAULT_VALUE, help_text=_(
        message='Sets the default state of the "Store results" checkbox.'
    )
)
