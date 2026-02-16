import functools
import logging

from django.apps import apps
from django.db.models.signals import m2m_changed, post_save, pre_delete
from django.utils.module_loading import import_string

from mayan.apps.common.utils import (
    ResolverPipelineModelAttribute, flatten_list, get_class_full_name
)

from .exceptions import DynamicSearchModelException, DynamicSearchQueryError
from .literals import MESSAGE_FEATURE_NO_STATUS
from .search_interpreters import SearchInterpreter
from .search_models import SearchModel
from .settings import (
    setting_backend, setting_backend_arguments, setting_results_limit
)

logger = logging.getLogger(name=__name__)


class SearchBackend:
    _initialized = False
    feature_reindex = False
    field_type_mapping = None

    @staticmethod
    def _disable():
        for search_model in SearchModel.all():
            post_save.disconnect(
                dispatch_uid='search_handler_index_instance',
                sender=search_model.model
            )
            pre_delete.disconnect(
                dispatch_uid='search_handler_deindex_instance',
                sender=search_model.model
            )

            for proxy in search_model.proxies:
                post_save.disconnect(
                    dispatch_uid='search_handler_index_instance',
                    sender=proxy
                )
                pre_delete.disconnect(
                    dispatch_uid='search_handler_deindex_instance',
                    sender=proxy
                )

            for related_model, path in search_model.get_related_models():
                post_save.disconnect(
                    dispatch_uid='search_handler_index_related_instance_{}_{}'.format(
                        get_class_full_name(klass=search_model.model),
                        get_class_full_name(klass=related_model)
                    ), sender=related_model
                )
                pre_delete.disconnect(
                    dispatch_uid='search_handler_index_related_instance_delete_{}_{}'.format(
                        get_class_full_name(klass=search_model.model),
                        get_class_full_name(klass=related_model)
                    ), sender=related_model
                )

        for through_model, data in SearchModel.get_through_models().items():
            m2m_changed.disconnect(
                dispatch_uid='search_handler_index_related_instance_m2m_{}'.format(
                    get_class_full_name(klass=through_model),
                ), sender=through_model
            )

    @staticmethod
    def _enable():
        # Hidden import.
        from .handlers import (
            handler_deindex_instance,
            handler_factory_index_related_instance_delete,
            handler_factory_index_related_instance_m2m,
            handler_factory_index_related_instance_save,
            handler_index_instance
        )

        for search_model in SearchModel.all():
            post_save.connect(
                dispatch_uid='search_handler_index_instance',
                receiver=handler_index_instance, sender=search_model.model
            )
            pre_delete.connect(
                dispatch_uid='search_handler_deindex_instance',
                receiver=handler_deindex_instance,
                sender=search_model.model, weak=False
            )

            for proxy in search_model.proxies:
                post_save.connect(
                    dispatch_uid='search_handler_index_instance',
                    receiver=handler_index_instance, sender=proxy
                )
                pre_delete.connect(
                    dispatch_uid='search_handler_deindex_instance',
                    receiver=handler_deindex_instance,
                    sender=proxy, weak=False
                )

            for related_model, path in search_model.get_related_models():
                post_save.connect(
                    dispatch_uid='search_handler_index_related_instance_{}_{}'.format(
                        get_class_full_name(klass=search_model.model),
                        get_class_full_name(klass=related_model)
                    ),
                    receiver=handler_factory_index_related_instance_save(
                        reverse_field_path=path
                    ), sender=related_model, weak=False
                )
                pre_delete.connect(
                    dispatch_uid='search_handler_index_related_instance_delete_{}_{}'.format(
                        get_class_full_name(klass=search_model.model),
                        get_class_full_name(klass=related_model)
                    ),
                    receiver=handler_factory_index_related_instance_delete(
                        reverse_field_path=path
                    ), sender=related_model, weak=False
                )

        for through_model, data in SearchModel.get_through_models().items():
            m2m_changed.connect(
                dispatch_uid='search_handler_index_related_instance_m2m_{}'.format(
                    get_class_full_name(klass=through_model)
                ),
                receiver=handler_factory_index_related_instance_m2m(
                    data=data
                ), sender=through_model, weak=False
            )

    @staticmethod
    def get_class():
        return import_string(dotted_path=setting_backend.value)

    @staticmethod
    def get_instance(extra_kwargs=None):
        kwargs = setting_backend_arguments.value.copy()
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        return SearchBackend.get_class()(**kwargs)

    @staticmethod
    def limit_queryset(queryset):
        pk_list = queryset.values('pk')[:setting_results_limit.value]
        return queryset.filter(pk__in=pk_list)

    @staticmethod
    def index_related_instance_m2m(
        action, instance, model, pk_set, search_model_related_paths
    ):
        # Hidden import
        from .tasks import task_index_instance

        if action in ('post_add', 'pre_remove'):
            instance_paths = search_model_related_paths.get(
                instance._meta.model, ()
            )
            model_paths = search_model_related_paths.get(
                model, ()
            )

            if action == 'pre_remove':
                exclude_kwargs = {
                    'exclude_app_label': instance._meta.app_label,
                    'exclude_model_name': instance._meta.model_name,
                    'exclude_kwargs': {'id': instance.pk}
                }
            else:
                exclude_kwargs = {}

            for instance_path in instance_paths:
                result = ResolverPipelineModelAttribute.resolve(
                    attribute=instance_path, obj=instance
                )

                try:
                    result = result.filter(pk__in=pk_set)
                except AttributeError:
                    """
                    Result is not a queryset. Exception can be safely
                    ignored.
                    """

                entries = flatten_list(value=result)

                for entry in entries:
                    task_kwargs = {
                        'app_label': entry._meta.app_label,
                        'model_name': entry._meta.model_name,
                        'object_id': entry.pk
                    }
                    task_kwargs.update(exclude_kwargs)

                    task_index_instance.apply_async(
                        kwargs=task_kwargs
                    )

            if action == 'pre_remove':
                exclude_kwargs = {
                    'exclude_app_label': model._meta.app_label,
                    'exclude_model_name': model._meta.model_name,
                    'exclude_kwargs': {'id__in': pk_set}
                }
            else:
                exclude_kwargs = {}

            for model_instance in model._meta.default_manager.filter(pk__in=pk_set):
                for instance_path in model_paths:
                    result = ResolverPipelineModelAttribute.resolve(
                        attribute=instance_path, obj=model_instance
                    )

                    entries = flatten_list(value=result)

                    for entry in entries:
                        task_kwargs = {
                            'app_label': entry._meta.app_label,
                            'model_name': entry._meta.model_name,
                            'object_id': entry.pk
                        }
                        task_kwargs.update(exclude_kwargs)

                        task_index_instance.apply_async(
                            kwargs=task_kwargs
                        )

    def __init__(self, _test_mode=False):
        self._test_mode = _test_mode

    def _search(
        self, search_field, query_type, value, is_quoted_value=False,
        is_raw_value=False
    ):
        raise NotImplementedError

    def deindex_instance(self, instance):
        """
        Optional method to remove an model instance from the search index.
        """

    def do_native_type_conversion(self, value):
        return value

    def do_query_type_verify(self, query_type, search_field):
        query_type_list = search_field.get_backend_field_query_type_list(
            search_backend=self
        )
        if query_type not in query_type_list:
            raise DynamicSearchQueryError(
                'The backend `{search_backend}` does not support queries '
                'of type `{query_type}` for the field named '
                '`{search_field}`.'.format(
                    query_type=query_type, search_backend=self,
                    search_field=search_field.field_name
                )
            )

    def get_field_type_mapping(self):
        return self.field_type_mapping or {}

    @functools.cache
    def get_resolved_field_type_map(self, search_model):
        """
        Return a dictionary that maps the search model search field class
        to the search engine's field class.
        """
        result = {}

        field_type_mapping = self.get_field_type_mapping()

        for search_field in search_model.search_fields:
            try:
                backend_field_type_dictionary = field_type_mapping[
                    search_field.field_class
                ]
            except KeyError:
                raise DynamicSearchModelException(
                    'Unknown field type `{}` for model `{}`'.format(
                        search_field.field_name, search_model.full_name
                    )
                )
            else:
                result[search_field.field_name] = backend_field_type_dictionary

        return result

    def get_search_field_backend_field_type(self, search_field):
        return self.get_resolved_field_type_map(
            search_model=search_field.search_model
        )[search_field.field_name]['field']

    def get_status(self):
        """
        Backend specific method to provide status and statistics information.
        """
        if not hasattr(self, '_get_status'):
            return MESSAGE_FEATURE_NO_STATUS
        else:
            return self._get_status()

    def index_instance(self, instance, exclude_model=None, exclude_kwargs=None):
        """
        Optional method to add or update an model instance to the search
        index.
        """

    def index_instances(self, search_model, id_list=None):
        """
        Optional method to add or update all instance of a model.
        """

    def initialize(self):
        self._initialize()

    def _initialize(self):
        """
        Optional method to setup the backend. Executed once on every boot up.
        """

    def refresh(self):
        """
        Forces all indexes to update and present an actual view of the
        search backend's objects and their values.
        """

    def reset(self, search_model=None):
        """
        Optional method to clear all search indices.
        """

    def search(
        self, query, search_model, user, store_resultset=False, queryset=None
    ):
        AccessControlList = apps.get_model(
            app_label='acls', model_name='AccessControlList'
        )
        SavedResultset = apps.get_model(
            app_label='dynamic_search', model_name='SavedResultset'
        )

        search_interpreter = SearchInterpreter.init(
            query=query, search_model=search_model
        )

        id_list = search_interpreter.do_resolve(search_backend=self)

        queryset = queryset or search_model.get_queryset()

        queryset = queryset.filter(pk__in=id_list)

        if search_model.permission:
            queryset = AccessControlList.objects.restrict_queryset(
                permission=search_model.permission, queryset=queryset,
                user=user
            )

        queryset = SearchBackend.limit_queryset(queryset=queryset)

        if store_resultset:
            search_explainer_text = search_interpreter.to_explain()

            saved_resultset = SavedResultset.objects.queryset_save(
                queryset=queryset,
                search_explainer_text=search_explainer_text,
                search_query=query, user=user
            )
        else:
            saved_resultset = None

        return (saved_resultset, queryset)

    def tear_down(self):
        """
        Optional method to clean up and/or destroy search backend structures
        like indices.
        """

    def test_mode_stop(self):
        """
        Optional method to cleanup after the tests end.
        """

    def upgrade(self):
        """
        Optional method to upgrade the search backend persistent structures.
        """
