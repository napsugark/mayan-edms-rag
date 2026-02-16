from django.contrib.contenttypes.models import ContentType
from django.core.files import File

from mayan.apps.acls.classes import ModelPermission
from mayan.apps.testing.tests.mixins import TestMixinObjectCreationTrack

from ..classes import Layer
from ..models import Asset, LayerTransformation
from ..tasks import task_content_object_image_generate
from ..transformations import BaseTransformation

from .literals import (
    TEST_ASSET_INTERNAL_NAME, TEST_ASSET_LABEL, TEST_ASSET_LABEL_EDITED,
    TEST_ASSET_PATH, TEST_LAYER_LABEL, TEST_LAYER_NAME, TEST_LAYER_ORDER,
    TEST_TRANSFORMATION_ARGUMENT, TEST_TRANSFORMATION_ARGUMENT_EDITED,
    TEST_TRANSFORMATION_LABEL, TEST_TRANSFORMATION_NAME
)


class AssetTestMixin(TestMixinObjectCreationTrack):
    _test_object_model = Asset
    _test_object_name = '_test_asset'

    def _create_test_asset(self):
        with open(file=TEST_ASSET_PATH, mode='rb') as file_object:
            self._test_asset = Asset.objects.create(
                label=TEST_ASSET_LABEL,
                internal_name=TEST_ASSET_INTERNAL_NAME,
                file=File(file=file_object)
            )


class AssetAPIViewTestMixin(AssetTestMixin):
    def _request_test_asset_create_api_view(self):
        self._test_object_track()

        with open(file=TEST_ASSET_PATH, mode='rb') as file_object:
            response = self.post(
                viewname='rest_api:asset-list', data={
                    'label': TEST_ASSET_LABEL,
                    'internal_name': TEST_ASSET_INTERNAL_NAME,
                    'file': File(file=file_object)
                }
            )

        self._test_object_set()

        return response

    def _request_test_asset_delete_api_view(self):
        return self.delete(
            viewname='rest_api:asset-detail',
            kwargs={'asset_id': self._test_asset.pk}
        )

    def _request_test_asset_detail_api_view(self):
        return self.get(
            viewname='rest_api:asset-detail',
            kwargs={'asset_id': self._test_asset.pk}
        )

    def _request_test_asset_edit_via_patch_api_view(self):
        return self.patch(
            viewname='rest_api:asset-detail', kwargs={
                'asset_id': self._test_asset.pk
            }, data={'label': TEST_ASSET_LABEL_EDITED}
        )

    def _request_test_asset_edit_via_put_api_view(self):
        with open(file=TEST_ASSET_PATH, mode='rb') as file_object:
            return self.put(
                viewname='rest_api:asset-detail', kwargs={
                    'asset_id': self._test_asset.pk
                }, data={
                    'label': TEST_ASSET_LABEL_EDITED,
                    'internal_name': TEST_ASSET_INTERNAL_NAME,
                    'file': File(file=file_object)
                }
            )

    def _request_test_asset_list_api_view(self):
        return self.get(viewname='rest_api:asset-list')


class AssetTaskTestMixin(AssetTestMixin):
    def _execute_asset_task_content_object_image_generate(self):
        content_type = ContentType.objects.get_for_model(model=Asset)

        task_content_object_image_generate.apply_async(
            kwargs={
                'content_type_id': content_type.pk,
                'object_id': self._test_asset.pk
            }
        ).get()


class AssetViewTestMixin(AssetTestMixin):
    def _request_test_asset_create_view(self):
        self._test_object_track()

        with open(file=TEST_ASSET_PATH, mode='rb') as file_object:
            response = self.post(
                viewname='converter:asset_create', data={
                    'label': TEST_ASSET_LABEL,
                    'internal_name': TEST_ASSET_INTERNAL_NAME,
                    'file': file_object
                }
            )

        self._test_object_set()

        return response

    def _request_test_asset_delete_view(self):
        return self.post(
            viewname='converter:asset_single_delete', kwargs={
                'asset_id': self._test_asset.pk
            }
        )

    def _request_test_asset_detail_view(self):
        return self.get(
            viewname='converter:asset_detail', kwargs={
                'asset_id': self._test_asset.pk
            }
        )

    def _request_test_asset_edit_view(self):
        return self.post(
            viewname='converter:asset_edit', kwargs={
                'asset_id': self._test_asset.pk
            }, data={
                'label': TEST_ASSET_LABEL_EDITED,
                'internal_name': TEST_ASSET_INTERNAL_NAME
            }
        )

    def _request_test_asset_list_view(self):
        return self.get(viewname='converter:asset_list')


class LayerTestCaseMixin:
    def setUp(self):
        super().setUp()
        Layer.invalidate_cache()


class LayerTestMixin:
    _test_transformation_object = None
    _test_transformation_object_parent = None
    auto_create_test_layer = True

    def setUp(self):
        super().setUp()

        self._test_transformation_object = self._test_document_version_page
        self._test_transformation_object_parent = self._test_document

        self._test_transformation_object_content_type = ContentType.objects.get_for_model(
            model=self._test_transformation_object
        )

        if self.auto_create_test_layer:
            self._create_test_permission()
            self._test_layer_permission_create = self._test_permission

            self._create_test_permission()
            self._test_layer_permission_delete = self._test_permission

            self._create_test_permission()
            self._test_layer_permission_edit = self._test_permission

            self._create_test_permission()
            self._test_layer_permission_view = self._test_permission

            self._test_layer = Layer(
                label=TEST_LAYER_LABEL, name=TEST_LAYER_NAME,
                order=TEST_LAYER_ORDER, permission_map={
                    'create': self._test_layer_permission_create,
                    'delete': self._test_layer_permission_delete,
                    'edit': self._test_layer_permission_edit,
                    'select': self._test_layer_permission_create,
                    'view': self._test_layer_permission_view
                }
            )

            ModelPermission.register(
                model=self._test_transformation_object_parent._meta.model,
                permissions=(
                    self._test_layer_permission_create,
                    self._test_layer_permission_delete,
                    self._test_layer_permission_edit,
                    self._test_layer_permission_view
                )
            )

    def tearDown(self):
        if self.auto_create_test_layer:
            Layer._registry.pop(self._test_layer.name, None)

        super().tearDown()


class TransformationTestMixin(LayerTestMixin, TestMixinObjectCreationTrack):
    _test_object_model = LayerTransformation
    _test_object_name = '_test_transformation'
    auto_create_test_transformation_class = True

    def setUp(self):
        super().setUp()
        if self.auto_create_test_transformation_class:
            self._create_test_transformation_class()

        BaseTransformation.register(
            layer=self._test_layer,
            transformation=self.TestTransformationClass
        )

    def _create_test_transformation(self):
        self._test_transformation = self._test_layer.add_transformation_to(
            obj=self._test_transformation_object,
            transformation_class=self.TestTransformationClass,
            arguments=getattr(
                self, '_test_transformation_arguments',
                TEST_TRANSFORMATION_ARGUMENT
            )
        )

    def _create_test_transformation_class(self):
        class TestTransformation(BaseTransformation):
            arguments = ('test_argument',)
            label = TEST_TRANSFORMATION_LABEL
            name = TEST_TRANSFORMATION_NAME

            def execute_on(self, *args, **kwargs):
                super().execute_on(*args, **kwargs)

                return self.image

        self.TestTransformationClass = TestTransformation


class TransformationViewTestMixin(TransformationTestMixin):
    def _request_transformation_create_post_view(self):
        self._test_object_track()

        response = self.post(
            viewname='converter:transformation_create', kwargs={
                'app_label': self._test_transformation_object_content_type.app_label,
                'model_name': self._test_transformation_object_content_type.model,
                'object_id': self._test_transformation_object.pk,
                'layer_name': self._test_layer.name,
                'transformation_name': self.TestTransformationClass.name
            }, data={
                'arguments': getattr(
                    self, '.test_transformation_argument',
                    TEST_TRANSFORMATION_ARGUMENT
                )
            }
        )

        self._test_object_set()

        return response

    def _request_transformation_create_get_view(self):
        return self.get(
            viewname='converter:transformation_create', kwargs={
                'app_label': self._test_transformation_object_content_type.app_label,
                'model_name': self._test_transformation_object_content_type.model,
                'object_id': self._test_transformation_object.pk,
                'layer_name': self._test_layer.name,
                'transformation_name': self.TestTransformationClass.name
            }, data={
                'arguments': getattr(
                    self, '.test_transformation_argument',
                    TEST_TRANSFORMATION_ARGUMENT
                )
            }
        )

    def _request_transformation_delete_multiple_view(self):
        return self.post(
            data={'id_list': self._test_transformation.pk},
            viewname='converter:transformation_delete_multiple', kwargs={
                'app_label': self._test_transformation_object_content_type.app_label,
                'model_name': self._test_transformation_object_content_type.model,
                'object_id': self._test_transformation_object.pk,
                'layer_name': self._test_layer.name
            }
        )

    def _request_transformation_delete_single_view(self):
        return self.post(
            viewname='converter:transformation_delete_single', kwargs={
                'app_label': self._test_transformation_object_content_type.app_label,
                'model_name': self._test_transformation_object_content_type.model,
                'object_id': self._test_transformation_object.pk,
                'layer_name': self._test_layer.name,
                'transformation_id': self._test_transformation.pk
            }
        )

    def _request_transformation_edit_view(self):
        return self.post(
            viewname='converter:transformation_edit', kwargs={
                'app_label': self._test_transformation_object_content_type.app_label,
                'model_name': self._test_transformation_object_content_type.model,
                'object_id': self._test_transformation_object.pk,
                'layer_name': self._test_layer.name,
                'transformation_id': self._test_transformation.pk
            }, data={
                'arguments': getattr(
                    self, '_test_transformation_argument_edited',
                    TEST_TRANSFORMATION_ARGUMENT_EDITED
                )
            }
        )

    def _request_transformation_list_view(self):
        return self.get(
            viewname='converter:transformation_list', kwargs={
                'app_label': self._test_transformation_object_content_type.app_label,
                'model_name': self._test_transformation_object_content_type.model,
                'object_id': self._test_transformation_object.pk,
                'layer_name': self._test_layer.name
            }
        )

    def _request_test_transformation_select_get_view(self):
        return self.get(
            viewname='converter:transformation_select', kwargs={
                'app_label': self._test_transformation_object_content_type.app_label,
                'model_name': self._test_transformation_object_content_type.model,
                'object_id': self._test_transformation_object.pk,
                'layer_name': self._test_layer.name
            }
        )

    def _request_test_transformation_select_post_view(self):
        return self.post(
            viewname='converter:transformation_select', kwargs={
                'app_label': self._test_transformation_object_content_type.app_label,
                'model_name': self._test_transformation_object_content_type.model,
                'object_id': self._test_transformation_object.pk,
                'layer_name': self._test_layer.name
            }, data={
                'transformation': self.TestTransformationClass.name
            }
        )
