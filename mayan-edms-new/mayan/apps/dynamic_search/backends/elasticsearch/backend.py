from collections import deque
import functools

import elasticsearch
from elasticsearch import Elasticsearch, helpers
from elasticsearch.dsl import Search

from mayan.settings.literals import DEFAULT_ELASTICSEARCH_PASSWORD

from ...exceptions import (
    DynamicSearchBackendException, DynamicSearchValueTransformationError
)
from ...search_backends import SearchBackend
from ...search_fields import SearchFieldVirtualAllFields
from ...search_models import SearchModel

from .literals import (
    DEFAULT_ELASTICSEARCH_HOSTS, DEFAULT_ELASTICSEARCH_INDICES_NAMESPACE,
    DEFAULT_ELASTICSEARCH_INDICES_NAMESPACE_TEST,
    DEFAULT_ELASTICSEARCH_POINT_IN_TIME_KEEP_ALIVE,
    DEFAULT_ELASTICSEARCH_SEARCH_PAGE_SIZE,
    DJANGO_TO_ELASTICSEARCH_FIELD_MAP, INDEX_NAME_DELIMITER,
    MAXIMUM_API_ATTEMPT_COUNT
)


class ElasticsearchSearchBackend(SearchBackend):
    feature_reindex = True
    field_type_mapping = DJANGO_TO_ELASTICSEARCH_FIELD_MAP

    def __init__(
        self, client_kwargs=None,
        indices_namespace=DEFAULT_ELASTICSEARCH_INDICES_NAMESPACE,
        search_page_size=DEFAULT_ELASTICSEARCH_SEARCH_PAGE_SIZE,
        point_in_time_keep_alive=DEFAULT_ELASTICSEARCH_POINT_IN_TIME_KEEP_ALIVE,
        **kwargs

    ):
        super().__init__(**kwargs)

        self.indices_namespace = indices_namespace
        self.point_in_time_keep_alive = point_in_time_keep_alive
        self.search_page_size = search_page_size

        self.client_kwargs = client_kwargs or {
            'basic_auth': ('elastic', DEFAULT_ELASTICSEARCH_PASSWORD),
            'hosts': DEFAULT_ELASTICSEARCH_HOSTS,
            'verify_certs': False
        }

        if self._test_mode:
            self.indices_namespace = DEFAULT_ELASTICSEARCH_INDICES_NAMESPACE_TEST

        self._client = Elasticsearch(**self.client_kwargs)

    def do_search_execute(self, index_name, search):
        model = self._get_model_for_index(index_name=index_name)
        self._client.indices.refresh(index=index_name)

        pk_field = model._meta.pk

        point_in_time = self._client.open_point_in_time(
            index=index_name, keep_alive=self.point_in_time_keep_alive
        )
        pit_id = point_in_time['id']

        base = (
            Search(using=self._client)
            .extra(
                pit={
                    'id': pit_id, 'keep_alive': self.point_in_time_keep_alive
                },
                size=self.search_page_size,
                track_total_hits=False
            )
            .sort('_shard_doc')
            .source(False)
        )

        query_dict = search.to_dict().get('query')
        if query_dict:
            base = base.update_from_dict(
                {'query': query_dict}
            )

        search_after = None

        try:
            while True:
                if search_after is None:
                    search_new = base
                else:
                    search_new = search_new.extra(search_after=search_after)

                response = search_new.execute()

                if not response or not len(response):
                    break

                for entry in response:
                    result_id = entry.meta.id
                    yield pk_field.to_python(value=result_id)

                search_after = response[-1].meta.sort
        except elasticsearch.exceptions.NotFoundError as exception:
            raise DynamicSearchBackendException(
                'Index not found. Make sure the search engine '
                'was properly initialized or upgraded if '
                'it already existed.'
            ) from exception
        finally:
            self._client.close_point_in_time(id=pit_id)

    def _get_index_name(self, search_model):
        return '{}{}{}'.format(
            self.indices_namespace, INDEX_NAME_DELIMITER,
            search_model.full_name
        )

    @functools.lru_cache(maxsize=256)
    def _get_model_for_index(self, index_name):
        indices_namespace, search_model_name = index_name.split(
            INDEX_NAME_DELIMITER
        )
        search_model = SearchModel.get(name=search_model_name)
        model = search_model.model

        return model

    @functools.cache
    def _get_search_model_index_mappings(self, search_model):
        mappings = {}

        field_map = self.get_resolved_field_type_map(
            search_model=search_model
        )
        for field_name, search_field_data in field_map.items():
            mappings[field_name] = {
                'type': search_field_data['field'].name
            }

            if 'analyzer' in search_field_data:
                mappings[field_name]['analyzer'] = search_field_data['analyzer']

        return mappings

    def _get_status(self):
        result = []

        title = 'Elasticsearch search model indexing status'
        title_length = len(title)
        result.append(title)
        result.append('=' * title_length)

        self.refresh()

        for search_model in SearchModel.all():
            index_name = self._get_index_name(search_model=search_model)
            try:
                index_stats = self._client.count(index=index_name)
            except elasticsearch.exceptions.NotFoundError:
                index_stats = {}

            count = index_stats.get('count', 'None')

            result.append(
                '{}: {}'.format(search_model.label, count)
            )

        return '\n'.join(result)

    def _initialize(self):
        self._update_mappings()

    def _search(
        self, search_field, query_type, value, is_quoted_value=False,
        is_raw_value=False
    ):
        self.do_query_type_verify(
            query_type=query_type, search_field=search_field
        )

        index_name = self._get_index_name(
            search_model=search_field.search_model
        )

        if isinstance(search_field, SearchFieldVirtualAllFields):
            seen = set()

            for search_field in search_field.field_composition:
                try:
                    search_field_query = query_type.resolve_for_backend(
                        is_quoted_value=is_quoted_value,
                        is_raw_value=is_raw_value, search_backend=self,
                        search_field=search_field, value=value
                    )
                except DynamicSearchValueTransformationError:
                    """Skip the search field."""
                else:
                    if search_field_query is not None:
                        index_name = self._get_index_name(
                            search_model=search_field.search_model
                        )

                        search = Search(index=index_name, using=self._client)
                        search = search.filter(search_field_query)

                        result = self.do_search_execute(
                            index_name=index_name, search=search
                        )
                        for item in result:
                            if item not in seen:
                                seen.add(item)
                                yield item
            else:
                return
        else:
            search = Search(index=index_name, using=self._client)

            try:
                search_field_query = query_type.resolve_for_backend(
                    is_quoted_value=is_quoted_value,
                    is_raw_value=is_raw_value, search_backend=self,
                    search_field=search_field, value=value
                )
            except DynamicSearchValueTransformationError:
                return
            else:
                if search_field_query is None:
                    return
                else:
                    search = search.filter(search_field_query)

                    yield from self.do_search_execute(
                        index_name=index_name, search=search
                    )

    def _update_mappings(self, search_model=None):
        if search_model:
            search_models = (search_model,)
        else:
            search_models = SearchModel.all()

        for search_model in search_models:
            index_name = self._get_index_name(search_model=search_model)

            try:
                self._client.indices.delete(index=index_name)
            except elasticsearch.exceptions.NotFoundError:
                """
                Non fatal, might be that this is the first time
                the method is executed. Proceed.
                """

            mappings = self._get_search_model_index_mappings(
                search_model=search_model
            )

            try:
                self._client.indices.create(
                    index=index_name,
                    mappings={'properties': mappings}
                )
            except elasticsearch.exceptions.RequestError:
                try:
                    self._client.indices.put_mapping(
                        index=index_name, properties=mappings
                    )
                except elasticsearch.exceptions.RequestError:
                    """
                    There are mapping changes that were not allowed.
                    Example: Text to Keyword.
                    Boot up regardless and allow user to reindex to delete
                    old indices.
                    """

    def deindex_instance(self, instance):
        search_model = SearchModel.get_for_model(instance=instance)
        index_name = self._get_index_name(search_model=search_model)
        self._client.delete(id=instance.pk, index=index_name)

    def index_instance(
        self, instance, exclude_model=None, exclude_kwargs=None
    ):
        search_model = SearchModel.get_for_model(instance=instance)

        document = search_model.populate(
            exclude_kwargs=exclude_kwargs, exclude_model=exclude_model,
            instance=instance, search_backend=self
        )
        index_name = self._get_index_name(search_model=search_model)
        self._client.index(
            document=document, id=instance.pk, index=index_name
        )

    def index_instances(self, search_model, id_list):
        index_name = self._get_index_name(search_model=search_model)

        def generate_actions():
            queryset = search_model.get_queryset()
            queryset = queryset.filter(pk__in=id_list)

            for instance in queryset:
                kwargs = search_model.populate(
                    search_backend=self, instance=instance
                )
                kwargs['_id'] = kwargs['id']

                yield kwargs

        action = generate_actions()
        bulk_indexing_generator = helpers.streaming_bulk(
            actions=action, client=self._client, index=index_name,
            yield_ok=False
        )

        deque(iterable=bulk_indexing_generator, maxlen=0)

    def refresh(self):
        attempt_count = 0
        search_model_index = 0
        search_models = SearchModel.all()

        while True:
            search_model = search_models[search_model_index]
            index_name = self._get_index_name(search_model=search_model)

            try:
                self._client.indices.refresh(index=index_name)
            except elasticsearch.exceptions.NotFoundError as exception:
                attempt_count += 1

                if attempt_count > MAXIMUM_API_ATTEMPT_COUNT:
                    raise DynamicSearchBackendException(
                        'Refresh attempt count exceeded the maximum'
                        ' of `{}`.'.format(
                            MAXIMUM_API_ATTEMPT_COUNT
                        )
                    ) from exception
            else:
                attempt_count = 0
                search_model_index += 1
                if search_model_index >= len(search_models):
                    break

    def reset(self, search_model=None):
        self.tear_down(search_model=search_model)
        self._update_mappings(search_model=search_model)

    def tear_down(self, search_model=None):
        if search_model:
            search_models = (search_model,)
        else:
            search_models = SearchModel.all()

        for search_model in search_models:
            index_name = self._get_index_name(search_model=search_model)
            try:
                self._client.indices.delete(index=index_name)
            except elasticsearch.exceptions.NotFoundError:
                """Ignore non existent indexes."""
