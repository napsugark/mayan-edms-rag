from django.db import models

from ...search_backends import SearchBackend
from ...search_models import SearchModel
from ...search_query_types import QueryType, QueryTypeExact

from ..backends import TestSearchBackendProxy
from ..literals import (
    TEST_OBJECT_BIG_INTEGER_VALUE, TEST_OBJECT_BOOLEAN_VALUE,
    TEST_OBJECT_CHAR_VALUE, TEST_OBJECT_EMAIL_VALUE,
    TEST_OBJECT_INTEGER_VALUE, TEST_OBJECT_POSITIVE_BIG_INTEGER_VALUE,
    TEST_OBJECT_POSITIVE_INTEGER_VALUE, TEST_OBJECT_TEXT_VALUE,
    TEST_OBJECT_UUID_VALUE
)


class SearchTestMixin:
    auto_test_search_backend_initialize = True
    auto_test_search_objects_create = False

    def _deindex_instance(self, instance):
        self._test_search_backend.deindex_instance(instance=instance)

    def _do_test_search(self, query):
        terms = str(
            tuple(
                query.values()
            )[0]
        ).strip()

        self.assertTrue(terms is not None)
        self.assertTrue(terms != '')

        return self._test_search_backend.search(
            search_model=self._test_search_model, query=query,
            user=self._test_case_user
        )

    def _index_instance(self, instance):
        self._test_search_backend.index_instance(instance=instance)

    def _setup_test_model_search(self):
        """
        This method allows tests to add model search configurations and
        not have to import and initialize the SearchBackend.
        """

    def setUp(self):
        self._existing_search_models = SearchModel._registry.copy()
        super().setUp()

        self._default_query_type_class = QueryType.get_default()
        QueryType.set_default(klass=QueryTypeExact)

        TestSearchBackendProxy._test_class = self.__class__
        self._test_search_backend = SearchBackend.get_instance()

        # Monkeypatch the search class so that the test behavior is only
        # enabled when called from a search test.
        self._setup_test_model_search()

        SearchBackend._enable()

        if self.auto_test_search_backend_initialize:
            self._test_search_backend._initialize()

        if self.auto_test_search_objects_create:
            self._create_test_search_objects()

        self._silence_logger(
            name='mayan.apps.dynamic_search.search_query_types'
        )

    def tearDown(self):
        self._test_search_backend.tear_down()
        self._test_search_backend.test_mode_stop()

        SearchBackend._disable()

        super().tearDown()
        QueryType.set_default(klass=self._default_query_type_class)
        SearchModel._registry = self._existing_search_models


class TestSearchObjectHierarchyTestMixin(SearchTestMixin):
    auto_test_search_objects_create = True

    def _create_test_models(self):
        self._create_test_model(
            fields={
                'label': models.CharField(
                    max_length=32
                )
            }, model_name='TestModelAttribute'
        )
        self._create_test_model(
            fields={
                'label': models.CharField(
                    max_length=32
                )
            }, model_name='TestModelGrandParent'
        )
        self._create_test_model(
            fields={
                'parent': models.ForeignKey(
                    on_delete=models.CASCADE, related_name='children',
                    to='TestModelGrandParent',
                ),
                'label': models.CharField(
                    max_length=32
                )
            }, model_name='TestModelParent'
        )
        self._create_test_model(
            fields={
                'parent': models.ForeignKey(
                    on_delete=models.CASCADE, related_name='children',
                    to='TestModelParent',
                ),
                'attributes': models.ManyToManyField(
                    related_name='children', to='TestModelAttribute',
                ),
                'label': models.CharField(
                    max_length=32
                )
            }, model_name='TestModelGrandChild'
        )
        self._create_test_model(
            base_class=self._test_model_dict['TestModelGrandChild'],
            model_name='TestModelGrandChildProxy',
            options={'proxy': True}
        )

    def _create_test_search_objects(self):
        self._test_object_grandparent = self._test_model_dict['TestModelGrandParent'].objects.create(
            label='grandparent'
        )
        self._test_object_parent = self._test_model_dict['TestModelParent'].objects.create(
            parent=self._test_object_grandparent,
            label='parent'
        )
        self._test_object_grandchild = self._test_model_dict['TestModelGrandChild'].objects.create(
            parent=self._test_object_parent,
            label='grandchild_0'
        )
        self._test_object_grandchildren = [self._test_object_grandchild]
        self._test_object_grandchild = self._test_model_dict['TestModelGrandChild'].objects.create(
            parent=self._test_object_parent,
            label='grandchild_1'
        )
        self._test_object_grandchildren.append(self._test_object_grandchild)
        self._test_object_grandchild_proxy = self._test_model_dict['TestModelGrandChildProxy'].objects.get(
            pk=self._test_object_grandchild.pk
        )

        self._test_object_attribute = self._test_model_dict['TestModelAttribute'].objects.create(
            label='attribute'
        )
        self._test_object_grandchild.attributes.add(
            self._test_object_attribute
        )

        self._test_object = self._test_object_attribute

    def _setup_test_model_search(self):
        self._test_search_grandparent = SearchModel(
            app_label=self._test_model_dict['TestModelGrandParent']._meta.app_label,
            model_name=self._test_model_dict['TestModelGrandParent']._meta.model_name
        )

        self._test_search_grandparent.add_model_field(field='label')
        self._test_search_grandparent.add_model_field(
            field='children__label'
        )
        self._test_search_grandparent.add_model_field(
            field='children__children__label'
        )
        self._test_search_grandparent.add_model_field(
            field='children__children__attributes__label'
        )

        self._test_search_grandchild = SearchModel(
            app_label=self._test_model_dict['TestModelGrandChild']._meta.app_label,
            model_name=self._test_model_dict['TestModelGrandChild']._meta.model_name
        )
        self._test_search_grandchild.add_model_field(
            field='label'
        )
        self._test_search_grandchild.add_model_field(
            field='attributes__label'
        )
        self._test_search_grandchild.add_proxy_model(
            app_label=self._test_model_dict['TestModelAttribute']._meta.app_label,
            model_name='TestModelGrandChildProxy'
        )

        self._test_search_attribute = SearchModel(
            app_label=self._test_model_dict['TestModelAttribute']._meta.app_label,
            model_name=self._test_model_dict['TestModelAttribute']._meta.model_name
        )
        self._test_search_attribute.add_model_field(
            field='label'
        )
        self._test_search_attribute.add_model_field(
            field='children__label'
        )

        self._test_search_model = self._test_search_attribute


class TestSearchObjectSimpleTestMixin(SearchTestMixin):
    _test_object_integer_set = True
    _test_object_biginteger_set = True
    auto_test_search_objects_create = True

    def _create_test_models(self):
        self._create_test_model(
            fields={
                'biginteger': models.BigIntegerField(blank=True, null=True),
                'boolean': models.BooleanField(blank=True, null=True),
                'char': models.CharField(
                    blank=True, max_length=32, null=True
                ),
                'datetime': models.DateTimeField(
                    auto_now_add=True, blank=True, null=True
                ),
                'email': models.EmailField(blank=True, null=True),
                'integer': models.IntegerField(blank=True, null=True),
                'positivebiginteger': models.PositiveBigIntegerField(
                    blank=True, null=True
                ),
                'positiveinteger': models.PositiveIntegerField(
                    blank=True, null=True
                ),
                'text': models.TextField(blank=True, null=True),
                'uuid': models.UUIDField(blank=True, null=True)
            }
        )

    def _create_test_search_objects(self):
        kwargs = {
            'boolean': TEST_OBJECT_BOOLEAN_VALUE,
            'char': TEST_OBJECT_CHAR_VALUE,
            'email': TEST_OBJECT_EMAIL_VALUE,
            'positivebiginteger': TEST_OBJECT_POSITIVE_BIG_INTEGER_VALUE,
            'positiveinteger': TEST_OBJECT_POSITIVE_INTEGER_VALUE,
            'text': TEST_OBJECT_TEXT_VALUE,
            'uuid': TEST_OBJECT_UUID_VALUE
        }

        if self._test_object_biginteger_set:
            kwargs['biginteger'] = TEST_OBJECT_BIG_INTEGER_VALUE

        if self._test_object_integer_set:
            kwargs['integer'] = TEST_OBJECT_INTEGER_VALUE

        self._test_object = self._test_model_dict['_TestModel_0'].objects.create(**kwargs)

    def _setup_test_model_search(self):
        self._test_search_model = SearchModel(
            app_label=self._test_model_dict['_TestModel_0']._meta.app_label,
            model_name=self._test_model_dict['_TestModel_0']._meta.model_name
        )

        self._test_search_model.add_model_field(field='biginteger')
        self._test_search_model.add_model_field(field='boolean')
        self._test_search_model.add_model_field(field='char')
        self._test_search_model.add_model_field(field='datetime')
        self._test_search_model.add_model_field(field='email')
        self._test_search_model.add_model_field(field='integer')
        self._test_search_model.add_model_field(field='positivebiginteger')
        self._test_search_model.add_model_field(field='positiveinteger')
        self._test_search_model.add_model_field(field='text')
        self._test_search_model.add_model_field(field='uuid')
