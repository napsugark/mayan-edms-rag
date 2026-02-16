import json
import unittest

from django.apps import apps
from django.db import connection, models

from rest_framework import status
from rest_framework.reverse import reverse

from mayan.apps.acls.classes import ModelPermission
from mayan.apps.databases.utils import get_model_ordering_fields
from mayan.apps.rest_api.api_view_mixins import ExternalObjectAPIViewMixin
from mayan.apps.testing.tests.base import GenericViewTestCase

from .. import generics, serializers

from .base import BaseAPITestCase
from .literals import TEST_OBJECT_LABEL, TEST_OBJECT_LABEL_EDITED
from .mixins import (
    DynamicFieldSerializerAPIViewTestCaseMixin, RESTAPIViewTestMixin
)


class RESTAPIViewTestCase(RESTAPIViewTestMixin, GenericViewTestCase):
    def test_browser_api_view(self):
        response = self._request_test_browser_api_view()
        self.assertEqual(response.status_code, 200)

    @unittest.skipIf(
        condition=connection.vendor != 'sqlite',
        reason='Skip for known Django issues #15802 and #27074'
    )
    def test_redoc_ui_view(self):
        response = self._request_test_redoc_ui_view()
        self.assertEqual(response.status_code, 200)

    @unittest.skipIf(
        condition=connection.vendor != 'sqlite',
        reason='Skip for known Django issues #15802 and #27074'
    )
    def test_swagger_ui_view(self):
        response = self._request_test_swagger_ui_view()
        self.assertEqual(response.status_code, 200)

    def test_swagger_no_ui_json_view(self):
        self.expected_content_types = ('application/json; charset=utf-8',)

        response = self._request_test_swagger_no_ui_json_view()
        self.assertEqual(response.status_code, 200)

    def test_swagger_no_ui_yaml_view(self):
        self.expected_content_types = ('application/yaml; charset=utf-8',)

        response = self._request_test_swagger_no_ui_yaml_view()
        self.assertEqual(response.status_code, 200)


class BatchAPIRequestViewTestCase(BaseAPITestCase):
    def setUp(self):
        super().setUp()

        self._create_test_permission()

        self._create_test_model(
            fields={
                'label': models.CharField(max_length=32, unique=True)
            }
        )

        ModelPermission.register(
            model=self._test_model_dict['_TestModel_0'], permissions=(self._test_permission,)
        )

        self._create_test_object(
            instance_kwargs={'label': TEST_OBJECT_LABEL}
        )

        class TestModelSerializer(serializers.ModelSerializer):
            class Meta:
                fields = ('id', 'label')
                model = self._test_model_dict['_TestModel_0']

        def _test_view_factory():
            class TestView(generics.ListCreateAPIView):
                mayan_object_permission_map = {'GET': self._test_permission}
                mayan_view_permission_map = {'POST': self._test_permission}
                serializer_class = TestModelSerializer
                source_queryset = self._test_model_dict['_TestModel_0'].objects.all()

            return TestView.as_view()

        self.add_test_view(
            test_view_factory=_test_view_factory,
            test_view_url=r'^test-view-url/$'
        )
        self._test_model_list_api_view_name = self._test_view_name

        def _test_view_factory():
            TestModel = self._test_model_dict['_TestModel_0']

            class TestView(generics.RetrieveUpdateDestroyAPIView):
                lookup_url_kwarg = 'test_object_id'
                mayan_object_permission_map = {
                    'DELETE': self._test_permission,
                    'GET': self._test_permission,
                    'PATCH': self._test_permission,
                    'PUT': self._test_permission
                }
                serializer_class = TestModelSerializer
                source_queryset = TestModel.objects.all()

            return TestView.as_view()

        self.add_test_view(
            test_view_factory=_test_view_factory,
            test_view_url=r'^test-view-url/(?P<test_object_id>\d+)/$'
        )
        self._test_model_detail_api_view_name = self._test_view_name

    def _request_batch_api_request_api_view(self, requests):
        return self.post(
            viewname='rest_api:batchrequest-create', data={
                'requests': requests
            }
        )

    def test_create_batch_api_request(self):
        self.grant_permission(permission=self._test_permission)

        requests = [
            {
                'body': {'label': TEST_OBJECT_LABEL},
                'method': 'POST',
                'name': 'test_request',
                'url': reverse(
                    viewname='rest_api:{}'.format(
                        self._test_model_list_api_view_name
                    )
                )
            }
        ]

        self._test_object.delete()

        test_model_count = self._test_model_dict['_TestModel_0'].objects.count()

        self._clear_events()

        response = self._request_batch_api_request_api_view(
            requests=json.dumps(obj=requests)
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 1
        )

        self.assertEqual(
            response.data['results'][0]['status_code'],
            status.HTTP_201_CREATED
        )

        self.assertEqual(
            self._test_model_dict['_TestModel_0'].objects.count(), test_model_count + 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_delete_batch_api_request(self):
        self.grant_access(
            obj=self._test_object, permission=self._test_permission
        )

        requests = [
            {
                'method': 'DELETE',
                'name': 'test_request',
                'url': reverse(
                    viewname='rest_api:{}'.format(
                        self._test_model_detail_api_view_name
                    ), kwargs={'test_object_id': self._test_object.pk}
                )
            }
        ]

        test_model_count = self._test_model_dict['_TestModel_0'].objects.count()

        self._clear_events()

        response = self._request_batch_api_request_api_view(
            requests=json.dumps(obj=requests)
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 1
        )

        self.assertEqual(
            response.data['results'][0]['status_code'],
            status.HTTP_204_NO_CONTENT
        )

        self.assertEqual(
            self._test_model_dict['_TestModel_0'].objects.count(), test_model_count - 1
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_edit_via_patch_batch_api_request(self):
        self.grant_access(
            obj=self._test_object, permission=self._test_permission
        )

        requests = [
            {
                'body': {'label': TEST_OBJECT_LABEL_EDITED},
                'method': 'PATCH',
                'name': 'test_request',
                'url': reverse(
                    viewname='rest_api:{}'.format(
                        self._test_model_detail_api_view_name
                    ), kwargs={'test_object_id': self._test_object.pk}
                )
            }
        ]

        test_model_label = self._test_object.label

        self._clear_events()

        response = self._request_batch_api_request_api_view(
            requests=json.dumps(obj=requests)
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 1
        )

        self.assertEqual(
            response.data['results'][0]['status_code'], status.HTTP_200_OK
        )

        self._test_object.refresh_from_db()
        self.assertNotEqual(self._test_object.label, test_model_label)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_edit_via_put_batch_api_request(self):
        self.grant_access(
            obj=self._test_object, permission=self._test_permission
        )

        requests = [
            {
                'body': {'label': TEST_OBJECT_LABEL_EDITED},
                'method': 'PUT',
                'name': 'test_request',
                'url': reverse(
                    viewname='rest_api:{}'.format(
                        self._test_model_detail_api_view_name
                    ), kwargs={'test_object_id': self._test_object.pk}
                )
            }
        ]

        test_model_label = self._test_object.label

        self._clear_events()

        response = self._request_batch_api_request_api_view(
            requests=json.dumps(obj=requests)
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 1
        )

        self.assertEqual(
            response.data['results'][0]['status_code'], status.HTTP_200_OK
        )

        self._test_object.refresh_from_db()
        self.assertNotEqual(self._test_object.label, test_model_label)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_list_get_batch_api_request(self):
        self.grant_access(
            obj=self._test_object, permission=self._test_permission
        )

        requests = [
            {
                'name': 'test_request',
                'url': reverse(
                    viewname='rest_api:{}'.format(
                        self._test_model_list_api_view_name
                    )
                )
            }
        ]

        self._clear_events()

        response = self._request_batch_api_request_api_view(
            requests=json.dumps(obj=requests)
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 1
        )

        self.assertEqual(
            response.data['results'][0]['status_code'], status.HTTP_200_OK
        )
        self.assertEqual(
            response.data['results'][0]['data']['count'], 1
        )
        self.assertEqual(
            response.data['results'][0]['data']['results'][0]['id'],
            self._test_object.pk
        )

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)

    def test_mass_edit_get_batch_api_request(self):
        self.grant_access(
            obj=self._test_object, permission=self._test_permission
        )

        requests = [
            {
                'name': 'test_object_list',
                'url': reverse(
                    viewname='rest_api:{}'.format(
                        self._test_model_list_api_view_name
                    )
                )
            },
            {
                'body': {'label': TEST_OBJECT_LABEL_EDITED},
                'iterables': ['test_object_list.data.results'],
                'method': 'PATCH',
                'name': 'test_object_edit',
                'url': '{}{{{{ iterables.0.id }}}}/'.format(
                    reverse(
                        viewname='rest_api:{}'.format(
                            self._test_model_list_api_view_name
                        )
                    )
                )
            }
        ]

        test_model_label = self._test_object.label

        self._clear_events()

        response = self._request_batch_api_request_api_view(
            requests=json.dumps(obj=requests)
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            response.data['count'], 2
        )

        self.assertEqual(
            response.data['results'][0]['status_code'], status.HTTP_200_OK
        )
        self.assertEqual(
            response.data['results'][0]['data']['count'], 1
        )
        self.assertEqual(
            response.data['results'][0]['data']['results'][0]['id'],
            self._test_object.pk
        )

        self.assertEqual(
            response.data['results'][1]['status_code'], status.HTTP_200_OK
        )

        self._test_object.refresh_from_db()
        self.assertNotEqual(self._test_object.label, test_model_label)

        events = self._get_test_events()
        self.assertEqual(events.count(), 0)


class DynamicFieldSerializerAPIViewTestCase(
    DynamicFieldSerializerAPIViewTestCaseMixin, BaseAPITestCase
):
    def _get_test_view_class(self, serializer_class):
        local_serializer_class = serializer_class

        class TestView(generics.RetrieveAPIView):
            lookup_url_kwarg = 'test_object_id'
            serializer_class = local_serializer_class
            source_queryset = self._test_model_dict['TestModelChild'].objects.all()

        return TestView

    def test_current_model_only_field_single(self):
        response = self._request_test_api_view(
            query={'_fields_only': 'test_field_3'}
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertTrue('parent' not in data)
        self.assertTrue('test_field_3' in data)
        self.assertTrue('test_field_4' not in data)

    def test_current_model_only_field_multiple(self):
        response = self._request_test_api_view(
            query={'_fields_only': 'test_field_3,test_field_4'}
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertTrue('parent' not in data)
        self.assertTrue('test_field_3' in data)
        self.assertTrue('test_field_4' in data)

    def test_current_model_only_rleated_field(self):
        response = self._request_test_api_view(
            query={'_fields_only': 'parent'}
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertTrue('parent' in data)
        self.assertTrue(
            'test_field_1' in data['parent']
        )
        self.assertTrue(
            'test_field_2' in data['parent']
        )
        self.assertTrue('test_field_3' not in data)
        self.assertTrue('test_field_4' not in data)

    def test_related_model_only_field_single(self):
        response = self._request_test_api_view(
            query={'_fields_only': 'parent__test_field_1'}
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertTrue('parent' in data)
        self.assertTrue(
            'test_field_1' in data['parent']
        )
        self.assertTrue(
            'test_field_2' not in data['parent']
        )
        self.assertTrue('test_field_3' not in data)
        self.assertTrue('test_field_4' not in data)

    def test_related_model_only_field_multiple(self):
        response = self._request_test_api_view(
            query={
                '_fields_only': 'parent__test_field_1,parent__test_field_2'
            }
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertTrue('parent' in data)
        self.assertTrue(
            'test_field_1' in data['parent']
        )
        self.assertTrue(
            'test_field_2' in data['parent']
        )
        self.assertTrue('test_field_3' not in data)
        self.assertTrue('test_field_4' not in data)

    def test_current_model_exclude_field_single(self):
        response = self._request_test_api_view(
            query={'_fields_exclude': 'test_field_3'}
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertTrue('parent' in data)
        self.assertTrue(
            'test_field_1' in data['parent']
        )
        self.assertTrue(
            'test_field_2' in data['parent']
        )
        self.assertTrue('test_field_3' not in data)
        self.assertTrue('test_field_4' in data)

    def test_current_model_exclude_field_multiple(self):
        response = self._request_test_api_view(
            query={'_fields_exclude': 'test_field_3,test_field_4'}
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertTrue('parent' in data)
        self.assertTrue(
            'test_field_1' in data['parent']
        )
        self.assertTrue(
            'test_field_2' in data['parent']
        )
        self.assertTrue('test_field_3' not in data)
        self.assertTrue('test_field_4' not in data)

    def test_current_model_exclude_related_field(self):
        response = self._request_test_api_view(
            query={'_fields_exclude': 'parent'}
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertTrue('parent' not in data)
        self.assertTrue('test_field_3' in data)
        self.assertTrue('test_field_4' in data)

    def test_related_model_exclude_field_single(self):
        response = self._request_test_api_view(
            query={'_fields_exclude': 'parent__test_field_1'}
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertTrue('parent' in data)
        self.assertTrue(
            'test_field_1' not in data['parent']
        )
        self.assertTrue(
            'test_field_2' in data['parent']
        )
        self.assertTrue('test_field_3' in data)
        self.assertTrue('test_field_4' in data)

    def test_related_model_exclude_field_multiple(self):
        response = self._request_test_api_view(
            query={
                '_fields_exclude': 'parent__test_field_1,parent__test_field_2'
            }
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertTrue('parent' in data)
        self.assertTrue(
            'test_field_1' not in data['parent']
        )
        self.assertTrue(
            'test_field_2' not in data['parent']
        )
        self.assertTrue('test_field_3' in data)
        self.assertTrue('test_field_4' in data)


class DynamicFieldSerializerWithMixinAPIViewTestCase(
    DynamicFieldSerializerAPIViewTestCaseMixin, BaseAPITestCase
):
    auto_add_test_view = True
    auto_create_test_object = False
    test_view_url = r'^test-view-url/(?P<test_object_id>\d+)/$'

    def _get_test_view_class(self, serializer_class):
        local_serializer_class = serializer_class

        class TestView(ExternalObjectAPIViewMixin, generics.RetrieveAPIView):
            external_object_queryset = self._test_model_dict['TestModelChild'].objects.all()
            external_object_pk_url_kwarg = 'test_object_id'
            lookup_url_kwarg = 'test_object_id'
            serializer_class = local_serializer_class
            source_queryset = self._test_model_dict['TestModelChild'].objects.all()

        return TestView

    def test_related_model_only_field_single_with_api_view_mixins(self):
        response = self._request_test_api_view(
            query={'_fields_only': 'parent__test_field_1'}
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertTrue('parent' in data)
        self.assertTrue(
            'test_field_1' in data['parent']
        )
        self.assertTrue(
            'test_field_2' not in data['parent']
        )
        self.assertTrue('test_field_3' not in data)
        self.assertTrue('test_field_4' not in data)


class ModelOrderingAPIViewTestCase(BaseAPITestCase):
    def _test_view_factory(self, test_model):
        class TestModelSerializer(serializers.ModelSerializer):
            class Meta:
                fields = (test_model._meta.pk.name,)
                model = test_model

        class TestAPIViewList(generics.ListAPIView):
            serializer_class = TestModelSerializer

            def get_source_queryset(self):
                manager = getattr(test_model, 'objects')
                return manager.all()

        return TestAPIViewList.as_view()

    def test_ordering_fields(self):
        models = apps.get_models()

        for model in models:
            ordering_field_list = get_model_ordering_fields(model=model)

            # Only test models with ordering fields beyond the primary key.
            if len(ordering_field_list) > 1:
                self.add_test_view(
                    test_view_factory_kwargs={'test_model': model}
                )
                viewname = 'rest_api:{}'.format(self._test_view_name)

                for ordering_field in ordering_field_list:
                    response = self.get(
                        query={'_ordering': ordering_field},
                        viewname=viewname
                    )
                    self.assertEqual(response.status_code, 200)

                self.remove_view_test()
