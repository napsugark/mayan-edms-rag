import logging

from django.template import Library

from ..literals import MESSAGE_SQLITE_WARNING
from ..utils import check_for_sqlite

logger = logging.getLogger(name=__name__)
register = Library()


@register.simple_tag(name='databases_check_sqlite')
def tag_databases_check_sqlite():
    if check_for_sqlite():
        return MESSAGE_SQLITE_WARNING
