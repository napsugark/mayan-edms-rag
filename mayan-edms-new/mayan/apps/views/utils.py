from django.urls import resolve as django_resolve
from django.urls.base import get_script_prefix

from .literals import URL_QUERY_POSITIVE_VALUES


def convert_to_id_list(items):
    return ','.join(
        map(str, items)
    )


def get_request_data(request):
    request_get_data = request.GET.dict()
    request_post_data = request.POST.dict()

    query_dict = request_get_data.copy()
    query_dict.update(request_post_data)

    return query_dict


def is_url_query_positive(value):
    if value is not None:
        return value.lower() in URL_QUERY_POSITIVE_VALUES


def request_is_ajax(request):
    return request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'


def resolve(path, urlconf=None):
    path = '/{}'.format(
        path.replace(
            get_script_prefix(), '', 1
        )
    )
    return django_resolve(path=path, urlconf=urlconf)
