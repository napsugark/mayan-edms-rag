import itertools
import logging

from django.apps import apps
from django.contrib.admin.utils import reverse_field_path
from django.db.models.aggregates import Max, Min
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
from django.utils.translation import gettext_lazy as _

from mayan.apps.common.class_mixins import AppsModuleLoaderMixin
from mayan.apps.common.utils import group_iterator, parse_range
from mayan.apps.databases.literals import DATABASE_MINIMUM_ID
from mayan.apps.views.literals import LIST_MODE_CHOICE_LIST

from .exceptions import DynamicSearchException
from .literals import QUERY_PARAMETER_ANY_FIELD
from .search_fields import SearchField
from .settings import (
    setting_indexing_chunk_size, setting_search_model_field_disable
)

logger = logging.getLogger(name=__name__)


class SearchModel(AppsModuleLoaderMixin):
    _loader_module_name = 'search'
    _registry = {}

    @staticmethod
    def function_return_same(value):
        return value

    @classmethod
    def all(cls):
        result = set(
            cls._registry.values()
        )
        result = list(result)
        result.sort(key=lambda entry: entry.label)
        return result

    @classmethod
    def get(cls, name):
        try:
            result = cls._registry[name]
        except KeyError:
            raise KeyError(
                _(message='Unknown search model `%s`.') % name
            )
        else:
            if getattr(result, 'serializer_path', None):
                result.serializer = import_string(
                    dotted_path=result.serializer_path
                )

        return result

    @classmethod
    def get_default(cls):
        for search_class in cls.all():
            if search_class.default:
                return search_class

    @classmethod
    def get_for_model(cls, instance):
        # Works the same for model classes and model instances.
        return cls.get(
            name=instance._meta.label.lower()
        )

    @classmethod
    def get_through_models(cls):
        through_models = {}

        for search_model in cls.all():
            for related_model, reverse_field_path_text in search_model.get_related_models():
                # Check is each related model is connected to a many to many.
                for field in related_model._meta.get_fields():
                    if field.many_to_many:
                        try:
                            through_model = field.through
                        except AttributeError:
                            through_model = field.remote_field.through

                        through_models.setdefault(
                            through_model, {}
                        )
                        through_models[through_model].setdefault(
                            related_model, set()
                        )
                        through_models[through_model][related_model].add(
                            reverse_field_path_text
                        )

        return through_models

    @classmethod
    def post_load_modules(cls):
        search_model_fields_disabled_dict_list = setting_search_model_field_disable.value

        for model_name, field_name_list in search_model_fields_disabled_dict_list.items():
            try:
                model = cls.get(name=model_name)
            except KeyError as exception:
                logger.error(
                    'Unable to load search model `%s` for field removal; %s',
                    model_name, exception
                )
            else:
                for field_name in field_name_list:
                    try:
                        search_field = model.get_search_field(
                            field_name=field_name
                        )
                    except DynamicSearchException as exception:
                        logger.error(
                            'Unable to remove field `%s` from search model '
                            '`%s`; %s', field_name, model_name, exception
                        )
                    else:
                        model.remove_search_field(search_field=search_field)

    def __init__(
        self, app_label, model_name, default=False, label=None,
        list_mode=None, manager_name=None, permission=None,
        queryset=None, serializer_path=None
    ):
        self.default = default
        self._label = label
        self.app_label = app_label
        self.list_mode = list_mode or LIST_MODE_CHOICE_LIST
        self.model_name = model_name.lower()
        self._proxies = []  # Lazy evaluation.
        self.permission = permission
        self.queryset = queryset
        self.search_fields_dict = {}
        self.serializer_path = serializer_path

        auto_field = self.base_model._meta.auto_field
        self.add_model_field(
            field=auto_field.name, label=auto_field.verbose_name
        )
        self.add_model_field(
            field=QUERY_PARAMETER_ANY_FIELD, label=_(message='All content')
        )

        self.manager_name = manager_name or self.model._meta.default_manager.name

        if default:
            for search_class in self.__class__._registry.values():
                search_class.default = False

        self.__class__._registry[self.full_name] = self

    def __repr__(self):
        return '<{}: {}>'.format(
            self.__class__.__name__, self.label
        )

    def __str__(self):
        return str(self.label)

    def add_model_field(self, **kwargs):
        """
        Add a search field that directly belongs to the parent SearchModel.
        """
        self.do_search_field_cache_invalidate()

        kwargs['search_model'] = self
        search_field = SearchField.init(**kwargs)
        self.search_fields_dict[search_field.field_name] = search_field
        return search_field

    def add_proxy_model(self, app_label, model_name):
        model_name = model_name.lower()
        self._proxies.append(
            {
                'app_label': app_label, 'model_name': model_name
            }
        )

        self.__class__._registry[
            '{}.{}'.format(app_label, model_name)
        ] = self

    @cached_property
    def base_model(self):
        return self.model._meta.proxy_for_model or self.model

    def do_search_field_cache_invalidate(self):
        self.__dict__.pop('search_field_name_list', None)
        self.__dict__.pop('search_fields', None)

    @cached_property
    def full_name(self):
        return '{}.{}'.format(self.app_label, self.model_name)

    full_name.short_description = _(message='Full name')

    def get_id_groups(self, range_string=None):
        """
        Generate ID groups when doing bulk indexing. ID groups avoid having
        to create a single task call for each object to be indexed.
        """
        queryset = self.model._meta.managers_map[self.manager_name].all()

        # Part 1 - Split the user requested range into blind groups.
        if not range_string:
            # If range is not specified it will be the minimum and maximum
            # IDs of the queryset.
            queryset_id_values = queryset.aggregate(
                min_id=Min('id'), max_id=Max('id')
            )
            range_string = '{}-{}'.format(
                queryset_id_values['min_id'] or DATABASE_MINIMUM_ID,
                queryset_id_values['max_id'] or DATABASE_MINIMUM_ID
            )

        # Part 2 - Validate the blind groups by querying them and retrieve
        # the valid ID values.
        id_list_groups = group_iterator(
            iterable=parse_range(range_string=range_string),
            group_size=setting_indexing_chunk_size.value
        )

        generator_valid_id_groups = (
            queryset.filter(
                pk__in=id_list
            ).values_list('id', flat=True) for id_list in id_list_groups
        )

        # Part 3 - Chain the valid ID groups into a single sequence and
        # split them again into groups.
        return group_iterator(
            iterable=itertools.chain.from_iterable(
                generator_valid_id_groups
            ), group_size=setting_indexing_chunk_size.value
        )

    def get_queryset(self):
        if self.queryset is not None:
            return self.queryset()
        else:
            return self.model._meta.managers_map[self.manager_name].all()

    def get_related_models(self):
        result = set()
        for search_field in self.search_fields:
            if search_field.concrete:
                obj, path = reverse_field_path(
                    model=self.model, path=search_field.field_name
                )
                if path:
                    # Ignore search model fields.
                    result.add(
                        (obj, path)
                    )

        return result

    def get_search_field(self, field_name):
        try:
            return self.search_fields_dict[field_name]
        except KeyError:
            raise DynamicSearchException(
                'No search field named: %s' % field_name
            )

    def get_search_field_choices(self):
        """
        Returns a list of the fields for the SearchModel.
        """
        result = []
        for search_field in self.search_fields:
            result.append(
                (search_field.field_name, search_field.label)
            )

        return sorted(
            result, key=lambda x: x[1]
        )

    @cached_property
    def label(self):
        if not self._label:
            self._label = self.model._meta.verbose_name
        return self._label

    @cached_property
    def model(self):
        return apps.get_model(
            app_label=self.app_label, model_name=self.model_name
        )

    @cached_property
    def pk(self):
        return self.full_name

    def populate(
        self, instance, search_backend, exclude_kwargs=None,
        exclude_model=None
    ):
        instance_field_data = {}

        # Process the search fields by order of priority. This makes sure
        # that virtual fields are processed last.
        for search_field in self.search_fields_priority_sorted:
            field_value = search_field.get_instance_value(
                exclude_kwargs=exclude_kwargs, exclude_model=exclude_model,
                instance=instance, instance_field_data=instance_field_data,
                search_backend=search_backend
            )
            if field_value is not None:
                instance_field_data[search_field.field_name] = field_value

        return instance_field_data

    @property
    def proxies(self):
        result = []
        for proxy in self._proxies:
            result.append(
                apps.get_model(
                    app_label=proxy['app_label'],
                    model_name=proxy['model_name']
                )
            )
        return result

    def remove_search_field(self, search_field):
        self.do_search_field_cache_invalidate()
        self.search_fields_dict.pop(search_field.field_name)

    @cached_property
    def search_field_name_list(self):
        return [
            search_field.field_name for search_field in self.search_fields
        ]

    @cached_property
    def search_fields(self):
        return self.search_fields_dict.values()

    @cached_property
    def search_fields_label_sorted(self):
        return sorted(
            self.search_fields,
            key=lambda search_field: search_field.get_label()
        )

    @cached_property
    def search_fields_priority_sorted(self):
        return sorted(
            self.search_fields,
            key=lambda search_field: search_field.priority
        )
