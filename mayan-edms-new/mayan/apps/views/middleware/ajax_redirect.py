from django.http import HttpResponseRedirect
from django.utils.deprecation import MiddlewareMixin

from mayan.apps.appearance.settings import setting_ajax_redirection_code

from ..utils import request_is_ajax


class AjaxRedirect(MiddlewareMixin):
    def process_request(self, request):
        ajax_referer = request.META.get('HTTP_X_ALT_REFERER')

        if ajax_referer:
            request.META['HTTP_REFERER'] = ajax_referer

        return None

    def process_response(self, request, response):
        if request_is_ajax(request=request):
            if isinstance(response, HttpResponseRedirect):
                response.status_code = setting_ajax_redirection_code.value
        return response
