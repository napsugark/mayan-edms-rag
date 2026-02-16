import random

from django.apps import apps
from django.contrib.contenttypes.models import ContentType
from django.db import connection, connections, models, transaction
from django.db.models.signals import post_save, pre_save
from django.db.utils import IntegrityError

from mayan.apps.acls.classes import ModelPermission
from mayan.apps.permissions.tests.mixins import PermissionTestMixin

from .utils import RandomSeedIdempotent


class ConnectionsCheckTestCaseMixin:
    _open_connections_check_enable = True

    def _get_open_connections_count(self):
        return len(
            connections.all()
        )

    def setUp(self):
        super().setUp()
        self._connections_count = self._get_open_connections_count()

    def tearDown(self):
        if self._open_connections_check_enable:

            self.assertEqual(
                self._connections_count, self._get_open_connections_count(),
                msg='Database connection leak. The number of database '
                'connections at the start and at the end of the test '
                'are not the same.'
            )

        super().tearDown()


class ContentTypeTestCaseMixin:
    def _inject_test_object_content_type(self):
        self._test_object_content_type = ContentType.objects.get_for_model(
            model=self._test_object
        )

        self._test_object_view_kwargs = {
            'app_label': self._test_object_content_type.app_label,
            'model_name': self._test_object_content_type.model,
            'object_id': self._test_object.pk
        }


class ModelTestCaseMixin:
    def _model_instance_to_dictionary(self, instance):
        return instance._meta.model._default_manager.filter(
            pk=instance.pk
        ).values()[0]


class PropertyModelTestViewMixin:
    def _request_model_property_list_view(self):
        return self.get(
            kwargs={
                'app_label': self._TestModel._meta.app_label,
                'model_name': self._TestModel._meta.model_name
            }, viewname='databases:model_property_list'
        )

    def _request_property_model_list_view(self):
        return self.get(viewname='databases:property_model_list')


class RandomPrimaryKeyModelMonkeyPatchMixin:
    random_primary_key_enable = True
    random_primary_key_maximum_attempts = 100
    random_primary_key_random_ceiling = 32767
    random_primary_key_random_floor = 1

    @classmethod
    def setUpClass(cls):
        RandomSeedIdempotent.seed()
        super().setUpClass()

    def do_patch_apply(self):
        if self.random_primary_key_enable:
            method_save_random_pk = self.factory_method_save_random_pk()

            if not getattr(models.Model, '_random_pk_patched', False):
                self._method_original_save = models.Model.save
                setattr(models.Model, 'save', method_save_random_pk)
                models.Model._random_pk_patched = True
            else:
                raise RuntimeError(
                    'The test class is already patched by '
                    '`RandomPrimaryKeyModelMonkeyPatchMixin`. This patch '
                    'must not be applied more than once. Fix the '
                    'previous test case to ensure `tearDown` is called.'
                )

    def do_patch_unapply(self):
        if self.random_primary_key_enable:
            models.Model.save = self._method_original_save
            models.Model._random_pk_patched = False

    def factory_method_save_random_pk(self):
        def method_save_random_pk(instance, *args, **kwargs):
            manager = instance._meta.default_manager

            if instance.pk:
                return self._method_original_save(
                    instance, *args, **kwargs
                )
            else:
                # Set meta.auto_created to True to have the original save_base
                # not send the pre_save signal which would normally send
                # the instance without a primary key. Since we assign a random
                # primary key any pre_save signal handler that relies on an
                # empty primary key will fail.
                # The meta.auto_created and manual pre_save sending emulates
                # the original behavior. Since meta.auto_created also disables
                # the post_save signal we must also send it ourselves.
                # This hack work with Django 1.11 .save_base() but can break
                # in future versions if that method is updated.
                pre_save.send(
                    sender=instance.__class__, instance=instance,
                    raw=False, update_fields=None
                )
                instance._meta.auto_created = True

                kwargs['force_insert'] = True

                attempts = 0
                while True:
                    primary_key_random = random.randint(
                        RandomPrimaryKeyModelMonkeyPatchMixin.random_primary_key_random_floor,
                        RandomPrimaryKeyModelMonkeyPatchMixin.random_primary_key_random_ceiling
                    )

                    instance.pk = primary_key_random

                    try:
                        with transaction.atomic():
                            result = self._method_original_save(
                                instance, *args, **kwargs
                            )
                    except IntegrityError:
                        queryset = manager.filter(pk=instance.pk).exists()
                        if queryset:
                            # Primary key collision.
                            attempts += 1

                            if attempts > RandomPrimaryKeyModelMonkeyPatchMixin.random_primary_key_maximum_attempts:
                                raise ValueError(
                                    'Maximum number of retries for a unique random primary key reached.'
                                )
                        else:
                            # This is not a primary key collision, re-raise
                            # the real problem.
                            raise
                    else:
                        instance._meta.auto_created = False

                        post_save.send(
                            created=True, instance=instance, raw=False,
                            sender=instance.__class__, update_fields=None
                        )

                        return result

        return method_save_random_pk

    def setUp(self):
        self.do_patch_apply()
        super().setUp()

    def tearDown(self):
        self.do_patch_unapply()
        super().tearDown()


class TestModelTestCaseMixin(ContentTypeTestCaseMixin, PermissionTestMixin):
    auto_create_test_object = False
    auto_create_test_object_fields = None
    auto_create_test_object_instance_kwargs = None
    auto_create_test_object_model = False
    auto_create_test_object_permission = False

    @staticmethod
    def _delete_test_model(model):
        ModelPermission.deregister(model=model)

        content_type = ContentType.objects.get_for_model(model=model)
        if content_type.pk:
            content_type.delete()

        # Only attempt to deleted if the model was not deleted as part
        # of the previous main test model deletion.
        if model in apps.all_models[model._meta.app_label] or model in apps.app_configs[model._meta.app_label].models:
            if not model._meta.proxy:
                with connection.schema_editor() as schema_editor:
                    schema_editor.delete_model(model=model)

        # Only attempt to delete if the model was not deleted as part
        # of the previous main test model deletion.
        TestModelTestCaseMixin._unregister_model(
            app_label=model._meta.app_label,
            model_name=model._meta.model_name
        )

        ContentType.objects.clear_cache()
        apps.clear_cache()

    @staticmethod
    def _unregister_model(app_label, model_name):
        try:
            del apps.all_models[app_label][model_name]
            del apps.app_configs[app_label].models[model_name]
        except KeyError:
            """
            Non fatal. Means the model was deleted in the first attempt.
            """

    @classmethod
    def setUpClass(cls):
        if connection.vendor == 'sqlite':
            connection.disable_constraint_checking()

        super().setUpClass()

    def _delete_test_models(self):
        # Delete the test models' content type entries and deregister the
        # permissions, this avoids their Content Type from being looked up
        # in subsequent tests where they don't exists due to the database
        # transaction rollback.

        # Clear previous model registration before re-registering it again to
        # avoid conflict with test models with the same name, in the same app
        # but from another test module.
        self._test_model_list.reverse()

        for model in self._test_model_list:
            model.objects.all().delete()

        for model in self._test_model_list:
            TestModelTestCaseMixin._delete_test_model(model=model)

        for model in self._test_model_list_extra:
            TestModelTestCaseMixin._delete_test_model(model=model)

    def setUp(self):
        self._test_model_dict = {}
        self._test_model_list = []
        self._test_model_list_extra = set()
        self._test_object_list = []

        super().setUp()

        if self.auto_create_test_object or self.auto_create_test_object_model:
            self._create_test_model(
                fields=self.auto_create_test_object_fields
            )

            if self.auto_create_test_object_permission:
                self._create_test_permission()

                ModelPermission.register(
                    model=self._TestModel, permissions=(
                        self._test_permission,
                    )
                )

        if self.auto_create_test_object:
            self._create_test_object(
                instance_kwargs=self.auto_create_test_object_instance_kwargs
            )

        self._create_test_models()

    def _create_test_models(self):
        """
        Optional test class method to have all of its test models created
        in the proper order.
        """

    def tearDown(self):
        self._delete_test_models()
        super().tearDown()

    def _create_test_model(
        self, base_class=None, fields=None, model_name=None,
        options=None
    ):
        base_class = base_class or models.Model
        test_model_count = len(self._test_model_list)
        self._test_model_name = model_name or '{}_{}'.format(
            '_TestModel', test_model_count
        )

        self.options = options
        # Obtain the app_config and app_label from the test's module path.
        self.app_config = apps.get_containing_app_config(
            object_name=self.__class__.__module__
        )

        if connection.vendor == 'mysql':
            self.skipTest(
                reason='MySQL doesn\'t support schema changes inside an '
                'atomic block.'
            )

        attrs = {
            '__module__': self.__class__.__module__,
            'Meta': self._get_test_model_meta()
        }

        if fields:
            attrs.update(fields)

        model_list_previous = set(
            apps.app_configs[self.app_config.label].models.values()
        )

        model = type(
            self._test_model_name, (base_class,), attrs
        )

        if not model._meta.proxy:
            with connection.schema_editor() as schema_editor:
                schema_editor.create_model(model=model)

        self._test_model_list_extra.update(
            set(
                apps.app_configs[self.app_config.label].models.values()
            ) - model_list_previous - {model}
        )

        self._TestModel = model
        self._test_model_dict[self._test_model_name] = model

        self._test_model_list.append(model)

        ContentType.objects.create(
            app_label=self._TestModel._meta.app_label,
            model=self._test_model_name.lower()
        )

        ContentType.objects.clear_cache()
        apps.clear_cache()

        return model

    def _create_test_object(self, instance_kwargs=None):
        instance_kwargs = instance_kwargs or {}

        if not getattr(self, '_TestModel', None):
            self._create_test_model()

        self._test_object = self._TestModel.objects.create(**instance_kwargs)
        self._inject_test_object_content_type()

        self._test_object_list.append(self._test_object)

    def _get_test_model_meta(self):
        self._test_db_table = '{}_{}'.format(
            self.app_config.label, self._test_model_name.lower()
        )

        class Meta:
            app_label = self.app_config.label
            db_table = self._test_db_table
            ordering = ('id',)
            verbose_name = self._test_model_name

        if self.options:
            for key, value in self.options.items():
                setattr(Meta, key, value)

        return Meta
