import itertools

from django.db import connection, models
from django.db.models import Q

from ...search_query_types import (
    BackendQueryType, QueryTypeExact, QueryTypeFuzzy, QueryTypeGreaterThan,
    QueryTypeGreaterThanOrEqual, QueryTypeLessThan, QueryTypeLessThanOrEqual,
    QueryTypePartial, QueryTypeRange, QueryTypeRangeExclusive,
    QueryTypeRegularExpression
)

from .backend import DjangoSearchBackend
from .literals import DEFAULT_FUZZY_SLOP, MAXIMUM_FUZZY_OPTIONS


class BackendQueryTypeExact(BackendQueryType):
    query_type = QueryTypeExact

    def do_resolve(self):
        if self.value is not None:
            if self.get_search_backend_field_type() == models.BooleanField:
                lookup_template = '{field_name}__exact'
                value_template = '{}'
            elif self.get_search_backend_field_type() == models.DateTimeField:
                backend_query_type = BackendQueryTypeRange(
                    is_quoted_value=self.is_quoted_value,
                    search_backend=self.search_backend,
                    search_field=self.search_field, value=(
                        self.value, self.value.replace(microsecond=999999)
                    ), extra_kwargs=self.extra_kwargs
                )
                return backend_query_type.do_resolve()
            elif self.get_search_backend_field_type() == models.UUIDField:
                lookup_template = '{field_name}_clean__exact'
                value_template = '{}'
            elif self.get_search_backend_field_type() == models.BigIntegerField:
                lookup_template = '{field_name}__exact'
                value_template = '{}'
            elif self.get_search_backend_field_type() == models.IntegerField:
                lookup_template = '{field_name}__exact'
                value_template = '{}'
            elif self.get_search_backend_field_type() == models.PositiveIntegerField:
                lookup_template = '{field_name}__exact'
                value_template = '{}'
            elif self.get_search_backend_field_type() == models.PositiveBigIntegerField:
                lookup_template = '{field_name}__exact'
                value_template = '{}'
            else:
                if self.is_quoted_value and self.value == '':
                    lookup_template = '{field_name}__exact'
                    value_template = '{}'
                else:
                    lookup_template = '{field_name}_clean__iregex'
                    if connection.vendor == 'postgresql':
                        value_template = r'\y{}\y'
                    else:
                        value_template = r'\b{}\b'

            return Q(
                **{
                    lookup_template.format(
                        field_name=self.search_field.field_name
                    ): value_template.format(self.value)
                }
            )


class BackendQueryFuzzy(BackendQueryType):
    query_type = QueryTypeFuzzy

    def do_resolve(self):
        fuzzy_options = []

        if self.value is not None:
            permutation_list = list(
                set(
                    [
                        ''.join(letters) for letters in itertools.permutations(self.value)
                    ]
                )
            )

            for permutation in permutation_list:
                difference_count = sum(1 for a, b in zip(self.value, permutation) if a != b)

                if difference_count <= DEFAULT_FUZZY_SLOP:
                    fuzzy_options.append(permutation)

            result = None
            for entry in fuzzy_options[:MAXIMUM_FUZZY_OPTIONS]:
                backend_query_type = BackendQueryTypeExact(
                    is_quoted_value=self.is_quoted_value,
                    search_backend=self.search_field,
                    search_field=self.search_field, value=entry,
                    extra_kwargs=self.extra_kwargs
                )

                query = backend_query_type.do_resolve()

                if result is None:
                    result = query
                else:
                    result |= query

            return result


class BackendQueryTypeGreaterThan(BackendQueryType):
    query_type = QueryTypeGreaterThan

    def do_resolve(self):
        if self.value is not None:
            return Q(
                **{
                    '{field_name}__gt'.format(
                        field_name=self.search_field.field_name
                    ): self.value
                }
            )


class BackendQueryTypeGreaterThanOrEqual(BackendQueryType):
    query_type = QueryTypeGreaterThanOrEqual

    def do_resolve(self):
        if self.value is not None:
            return Q(
                **{
                    '{field_name}__gte'.format(
                        field_name=self.search_field.field_name
                    ): self.value
                }
            )


class BackendQueryTypeLessThan(BackendQueryType):
    query_type = QueryTypeLessThan

    def do_resolve(self):
        if self.value is not None:
            return Q(
                **{
                    '{field_name}__lt'.format(
                        field_name=self.search_field.field_name
                    ): self.value
                }
            )


class BackendQueryTypeLessThanOrEqual(BackendQueryType):
    query_type = QueryTypeLessThanOrEqual

    def do_resolve(self):
        if self.value is not None:
            return Q(
                **{
                    '{field_name}__lte'.format(
                        field_name=self.search_field.field_name
                    ): self.value
                }
            )


class BackendQueryTypePartial(BackendQueryType):
    query_type = QueryTypePartial

    def do_resolve(self):
        if self.value is not None:
            return Q(
                **{
                    '{field_name}_clean__icontains'.format(
                        field_name=self.search_field.field_name
                    ): self.value
                }
            )


class BackendQueryTypeRange(BackendQueryType):
    query_type = QueryTypeRange

    def do_resolve(self):
        if self.value is not None:
            return Q(
                **{
                    '{field_name}__gte'.format(
                        field_name=self.search_field.field_name
                    ): self.value[0], '{field_name}__lte'.format(
                        field_name=self.search_field.field_name
                    ): self.value[1]
                }
            )


class BackendQueryTypeRangeExclusive(BackendQueryType):
    query_type = QueryTypeRangeExclusive

    def do_resolve(self):
        if self.value is not None:
            return Q(
                **{
                    '{field_name}__gt'.format(
                        field_name=self.search_field.field_name
                    ): self.value[0], '{field_name}__lt'.format(
                        field_name=self.search_field.field_name
                    ): self.value[1]
                }
            )


class BackendQueryTypeRegularExpression(BackendQueryType):
    query_type = QueryTypeRegularExpression

    def do_resolve(self):
        if self.value is not None:
            return Q(
                **{
                    '{field_name}_clean__regex'.format(
                        field_name=self.search_field.field_name
                    ): self.value
                }
            )


BackendQueryType.register(
    klass=BackendQueryTypeExact, search_backend=DjangoSearchBackend
)
BackendQueryType.register(
    klass=BackendQueryFuzzy, search_backend=DjangoSearchBackend
)
BackendQueryType.register(
    klass=BackendQueryTypeGreaterThan, search_backend=DjangoSearchBackend
)
BackendQueryType.register(
    klass=BackendQueryTypeGreaterThanOrEqual,
    search_backend=DjangoSearchBackend
)
BackendQueryType.register(
    klass=BackendQueryTypeLessThan, search_backend=DjangoSearchBackend
)
BackendQueryType.register(
    klass=BackendQueryTypeLessThanOrEqual,
    search_backend=DjangoSearchBackend
)
BackendQueryType.register(
    klass=BackendQueryTypePartial, search_backend=DjangoSearchBackend
)
BackendQueryType.register(
    klass=BackendQueryTypeRange,
    search_backend=DjangoSearchBackend
)
BackendQueryType.register(
    klass=BackendQueryTypeRangeExclusive,
    search_backend=DjangoSearchBackend
)
BackendQueryType.register(
    klass=BackendQueryTypeRegularExpression,
    search_backend=DjangoSearchBackend
)
