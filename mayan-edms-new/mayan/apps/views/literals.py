from django.utils.translation import gettext_lazy as _

DEFAULT_VIEWS_PAGINATE_BY = 30
DEFAULT_VIEWS_PAGING_ARGUMENT = 'page'

LIST_MODE_CHOICE_LIST = 'list'
LIST_MODE_CHOICE_ITEM = 'items'

LIST_MODE_CHOICES = (
    (LIST_MODE_CHOICE_ITEM, _(message='Items')),
    (LIST_MODE_CHOICE_LIST, _(message='List'))
)

PK_LIST_KEY = 'id_list'
PK_LIST_SEPARATOR = ','

TEST_SERVER_HOST = 'testserver'
TEST_SERVER_SCHEME = 'http'
TEST_VIEW_NAME = 'test_view_name'
TEST_VIEW_URL = 'test-view-url'

TEXT_LIST_AS_ITEMS_PARAMETER = '_list_mode'
TEXT_LIST_AS_ITEMS_VARIABLE_NAME = 'list_as_items'

TEXT_SORT_FIELD_PARAMETER = '_ordering'
TEXT_SORT_FIELD_VARIABLE_NAME = 'ordering'

URL_QUERY_POSITIVE_VALUES = ('on', 'true', 'yes')
