from mayan.apps.icons.icons import Icon

icon_announcement_create = icon_tag_create = Icon(
    driver_name='fontawesome-dual', primary_symbol='bullhorn',
    secondary_symbol='plus'
)
icon_announcement_delete_multiple = Icon(
    driver_name='fontawesome', symbol='times'
)
icon_announcement_delete_single = icon_announcement_delete_multiple
icon_announcement_edit = Icon(driver_name='fontawesome', symbol='pencil-alt')
icon_announcement_list = Icon(driver_name='fontawesome', symbol='bullhorn')
