import importlib

from furl import furl

from django.conf import settings
from django.http import HttpResponse
from django.http.response import FileResponse
from django.template import Context, Template
from django.test.utils import ContextList
from django.urls import clear_url_caches, re_path, resolve, reverse
from django.utils.encoding import force_bytes

from stronghold.decorators import public

from ..literals import (
    TEST_SERVER_HOST, TEST_SERVER_SCHEME, TEST_VIEW_NAME, TEST_VIEW_URL
)
from ..models import UserConfirmView


class ClientMethodsTestCaseMixin:
    def _build_verb_kwargs(self, viewname=None, path=None, *args, **kwargs):
        data = kwargs.pop('data', None) or {}
        follow = kwargs.pop('follow', False)
        query = kwargs.pop('query', None) or {}
        headers = kwargs.pop('headers', None) or {}

        if viewname:
            path = reverse(viewname=viewname, *args, **kwargs)

        path = furl(url=path)
        path.args.update(query)

        result = {
            'follow': follow, 'data': data, 'path': path.tostr()
        }
        result.update(headers)
        return result

    def delete(self, viewname=None, path=None, *args, **kwargs):
        return self.client.delete(
            **self._build_verb_kwargs(
                path=path, viewname=viewname, *args, **kwargs
            )
        )

    def generic(self, method, viewname=None, path=None, *args, **kwargs):
        return self.client.generic(
            method=method, **self._build_verb_kwargs(
                path=path, viewname=viewname, *args, **kwargs
            )
        )

    def get(self, viewname=None, path=None, *args, **kwargs):
        return self.client.get(
            **self._build_verb_kwargs(
                path=path, viewname=viewname, *args, **kwargs
            )
        )

    def patch(self, viewname=None, path=None, *args, **kwargs):
        return self.client.patch(
            **self._build_verb_kwargs(
                path=path, viewname=viewname, *args, **kwargs
            )
        )

    def post(self, viewname=None, path=None, *args, **kwargs):
        return self.client.post(
            **self._build_verb_kwargs(
                path=path, viewname=viewname, *args, **kwargs
            )
        )

    def put(self, viewname=None, path=None, *args, **kwargs):
        return self.client.put(
            **self._build_verb_kwargs(
                path=path, viewname=viewname, *args, **kwargs
            )
        )


class DownloadTestCaseMixin:
    def assert_download_response(
        self, response, content=None, filename=None, is_attachment=None,
        mime_type=None
    ):
        self.assertTrue(
            isinstance(response, FileResponse)
        )

        if filename:
            self.assertEqual(response.filename, filename)

        if content:
            response_content = b''.join(
                [
                    force_bytes(s=block) for block in response
                ]
            )
            self.assertEqual(response_content, content)

        if is_attachment is not None:
            self.assertTrue(response.as_attachment)

        if mime_type:
            self.assertTrue(
                response.headers['Content-Type'].startswith(mime_type)
            )


class TestServerTestCaseMixin:
    def setUp(self):
        super().setUp()
        self.testserver_prefix = self.get_testserver_prefix()
        self.testserver_url = self.get_testserver_url()
        self.test_view_request = None

    def _test_view_factory(self, test_object=None):
        def test_view(request):
            self.test_view_request = request
            return HttpResponse()

        return test_view

    def get_testserver_prefix(self):
        return furl(
            scheme=TEST_SERVER_SCHEME, host=TEST_SERVER_HOST,
        ).tostr()

    def get_testserver_url(self):
        return furl(
            scheme=TEST_SERVER_SCHEME, host=TEST_SERVER_HOST,
            path=self.test_view_url
        ).tostr()


class TestViewTestCaseMixin:
    auto_add_test_view = False
    test_view_is_public = False
    test_view_object = None
    test_view_name = None
    test_view_template = '{{ object }}'
    test_view_url = TEST_VIEW_URL

    def setUp(self):
        super().setUp()
        self._test_view_count = 0
        self._test_view_name_list = []
        self._test_view_url_pattern_list = []

        if self.auto_add_test_view:
            self.add_test_view(test_object=self.test_view_object)

    def tearDown(self):
        self.client.logout()

        for _ in range(self._test_view_count):
            self.remove_view_test()

        super().tearDown()

    def _get_context_from_test_response(self, response):
        if isinstance(response.context, ContextList):
            # Template widget rendering causes test client response to be
            # `ContextList` rather than `RequestContext`. Typecast to
            # dictionary before updating.
            result = dict(response.context).copy()
            result.update(
                {'request': response.wsgi_request}
            )
            context = Context(result)
        else:
            result = response.context or {}
            result.update(
                {'request': response.wsgi_request}
            )
            context = Context(result)

        context.request = response.wsgi_request
        return context

    def _get_test_view_urlpatterns(self):
        root_url_module = importlib.import_module(name=settings.ROOT_URLCONF)

        return root_url_module.urlpatterns

    def _test_view_factory(self, test_object=None):
        def test_view(request):
            template = Template(template_string=self.test_view_template)
            context = Context(
                dict_={'object': test_object, 'resolved_object': test_object}
            )
            return HttpResponse(
                content=template.render(context=context)
            )

        if self.test_view_is_public:
            return public(function=test_view)
        else:
            return test_view

    def add_test_view(
        self, test_object=None, test_view_factory=None, test_view_name=None,
        test_view_url=None, test_view_factory_kwargs=None
    ):
        test_view_factory_kwargs = test_view_factory_kwargs or {}

        # For compatibility with the previous interface.
        if test_object:
            test_view_factory_kwargs['test_object'] = test_object

        if test_view_factory:
            view = test_view_factory(**test_view_factory_kwargs)
        else:
            view = self._test_view_factory(**test_view_factory_kwargs)

        if test_view_name:
            self._test_view_name = test_view_name
        else:
            self._test_view_name = '{}_{}'.format(
                TEST_VIEW_NAME, self._test_view_count
            )

        url_pattern = re_path(
            route=test_view_url or self.test_view_url, view=view,
            name=self._test_view_name
        )

        self._test_view_url_pattern_list.append(url_pattern)

        self._get_test_view_urlpatterns().insert(0, url_pattern)
        clear_url_caches()

        self._test_view_count += 1
        self._test_view_name_list.append(self._test_view_name)

    def get_test_view(self):
        response = self.get(viewname=self._test_view_name)
        return self._get_context_from_test_response(response=response)

    def remove_view_test(self):
        self._test_view_count -= 1
        self._test_view_name = self._test_view_name_list.pop(
            self._test_view_count
        )
        url_pattern = self._test_view_url_pattern_list.pop(
            self._test_view_count
        )

        app_urlpatterns = self._get_test_view_urlpatterns()

        index = app_urlpatterns.index(url_pattern)
        self._get_test_view_urlpatterns().pop(index)


class UserConfirmViewPropertyTestMixin:
    _auto_create_test_user_confirm_view_property = True
    _test_user_confirm_view_property_kwargs = {'user_confirmation_view_id': 1}
    _test_user_confirm_view_property_viewname = 'views:user_views_confirm_delete_single'

    def setUp(self):
        super().setUp()

        if self._auto_create_test_user_confirm_view_property:
            self._create_test_user_confirm_view_property()

    def _create_test_user_confirm_view_property(self):
        path = reverse(
            kwargs=self._test_user_confirm_view_property_kwargs,
            viewname=self._test_user_confirm_view_property_viewname
        )

        resolver_match = resolve(path=path)

        view_name = '{}:{}'.format(
            resolver_match.namespace, resolver_match.url_name
        )

        self._test_confirm_view, created = UserConfirmView.objects.update_or_create(
            defaults={
                'namespace': resolver_match.namespace,
                'remember': True
            }, name=view_name, user=self._test_case_user
        )


class UserConfirmViewPropertyViewTestMixin(UserConfirmViewPropertyTestMixin):
    def _request_user_confirm_property_delete_view(self, remember=None):
        return self.post(
            kwargs={'user_confirmation_view_id': self._test_confirm_view.pk},
            viewname='views:user_views_confirm_delete_single'
        )

    def _request_user_confirm_property_list_view(self):
        return self.get(
            kwargs={'user_id': self._test_case_user.pk},
            viewname='views:user_views_confirm_list'
        )


class UserViewModeViewTestMixin:
    def _request_test_current_user_view_modes_view(self):
        return self._request_test_user_view_modes_view(
            user=self._test_case_user
        )

    def _request_test_super_user_view_modes_view(self):
        return self._request_test_user_view_modes_view(
            user=self._test_super_user
        )

    def _request_test_user_view_modes_view(self, user=None):
        user = user or self._test_user

        return self.get(
            viewname='views:user_view_modes', kwargs={'user_id': user.pk}
        )
