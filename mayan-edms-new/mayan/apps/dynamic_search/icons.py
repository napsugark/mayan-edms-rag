from mayan.apps.icons.icons import Icon

icon_result_list = Icon(driver_name='fontawesome', symbol='search')

icon_saved_resultset_delete_single = Icon(
    driver_name='fontawesome', symbol='times'
)
icon_saved_resultset_list = Icon(
    driver_name='fontawesome', symbol='diagram-next'
)
icon_saved_resultset_result_list = Icon(
    driver_name='fontawesome', symbol='diagram-next'
)

icon_search = Icon(driver_name='fontawesome', symbol='search')
icon_search_advanced = Icon(driver_name='fontawesome', symbol='search-plus')
icon_search_again = Icon(driver_name='fontawesome', symbol='sync')
icon_search_backend_reindex = Icon(
    driver_name='fontawesome-layers', data=[
        {
            'class': 'fas fa-circle',
            'transform': 'down-3 right-10',
            'mask': 'fas fa-search'
        },
        {'class': 'far fa-circle', 'transform': 'down-3 right-10'},
        {'class': 'fas fa-search', 'transform': 'flip-h left-3'},
        {'class': 'fas fa-hammer', 'transform': 'shrink-4 down-3 right-10'}
    ]
)
icon_search_form_clear = Icon(
    driver_name='fontawesome', symbol='times-circle'
)
icon_search_model_detail = Icon(
    driver_name='fontawesome', symbol='square-binary'
)
icon_search_model_list = Icon(
    driver_name='fontawesome', symbol='file-invoice'
)
icon_search_submit = Icon(driver_name='fontawesome', symbol='search')
