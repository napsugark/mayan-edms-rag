from mayan.apps.icons.icons import Icon

# App

icon_document_file_content = Icon(driver_name='fontawesome', symbol='font')

# Document file

icon_document_file_content_delete_multiple = Icon(
    driver_name='fontawesome-dual', primary_symbol='font',
    secondary_symbol='times'
)
icon_document_file_content_delete_single = icon_document_file_content_delete_multiple
icon_document_file_content_detail = Icon(
    driver_name='fontawesome', symbol='font'
)
icon_document_file_content_download = Icon(
    driver_name='fontawesome-dual', primary_symbol='font',
    secondary_symbol='arrow-down'
)
icon_document_file_parsing_submit_multiple = Icon(
    driver_name='fontawesome-dual', primary_symbol='font',
    secondary_symbol='arrow-right'
)
icon_document_file_parsing_submit_single = icon_document_file_parsing_submit_multiple

# Document file page

icon_document_file_page_content_detail = icon_document_file_content

# Document type

icon_document_type_parsing_settings = Icon(
    driver_name='fontawesome', symbol='font'
)
icon_document_type_parsing_submit = icon_document_file_parsing_submit_multiple
