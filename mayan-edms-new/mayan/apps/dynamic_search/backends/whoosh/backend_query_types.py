import whoosh
from whoosh.qparser import FuzzyTermPlugin, GtLtPlugin, RegexPlugin
from whoosh.qparser.dateparse import DateParserPlugin

from ...search_query_types import (
    BackendQueryType, QueryTypeExact, QueryTypeFuzzy, QueryTypeGreaterThan,
    QueryTypeGreaterThanOrEqual, QueryTypeLessThan, QueryTypeLessThanOrEqual,
    QueryTypePartial, QueryTypeRange, QueryTypeRangeExclusive,
    QueryTypeRegularExpression
)

from .backend import WhooshSearchBackend


class BackendQueryTypeWhoosh(BackendQueryType):
    def do_resolve(self):
        if self.get_search_backend_field_type() == whoosh.fields.DATETIME:

            self.extra_kwargs['parser'].add_plugin(
                DateParserPlugin()
            )

        return self._do_resolve()


class BackendQueryTypeExact(BackendQueryTypeWhoosh):
    query_type = QueryTypeExact

    def _do_resolve(self):
        if self.value is not None:
            if not self.value and self.is_quoted_value:
                return 'NOT *'
            else:
                if self.is_quoted_value:
                    template = '{}:("{}")'
                else:
                    template = '{}:({})'

                return template.format(
                    self.search_field.field_name, self.value
                )


class BackendQueryTypeFuzzy(BackendQueryTypeWhoosh):
    query_type = QueryTypeFuzzy

    def _do_resolve(self):
        self.extra_kwargs['parser'].add_plugin(
            FuzzyTermPlugin()
        )

        if self.value is not None:
            if self.is_quoted_value:
                template = '{}:"{}"~2'
            else:
                template = '{}:{}~2'

            return template.format(
                self.search_field.field_name, self.value
            )


class BackendQueryTypeComparison(BackendQueryTypeWhoosh):
    def _do_resolve(self):
        self.extra_kwargs['parser'].add_plugin(
            GtLtPlugin()
        )

        if self.value is not None:
            return self.template.format(
                self.search_field.field_name, self.value
            )


class BackendQueryTypeGreaterThan(BackendQueryTypeComparison):
    query_type = QueryTypeGreaterThan
    template = '{}:>{}'


class BackendQueryTypeGreaterThanOrEqual(BackendQueryTypeComparison):
    query_type = QueryTypeGreaterThanOrEqual
    template = '{}:>={}'


class BackendQueryTypeLessThan(BackendQueryTypeComparison):
    query_type = QueryTypeLessThan
    template = '{}:<{}'


class BackendQueryTypeLessThanOrEqual(BackendQueryTypeComparison):
    query_type = QueryTypeLessThanOrEqual
    template = '{}:<={}'


class BackendQueryTypePartial(BackendQueryTypeWhoosh):
    query_type = QueryTypePartial

    def _do_resolve(self):
        if self.value is not None:
            if self.is_quoted_value:
                template = '{}:"{}"'
            else:
                template = '{}:(*{}*)'

            if self.get_search_backend_field_type() == whoosh.fields.BOOLEAN:
                template = '{}:({})'

            if self.value is not None:
                return template.format(
                    self.search_field.field_name, self.value
                )


class BackendQueryTypeRange(BackendQueryTypeWhoosh):
    query_type = QueryTypeRange

    def _do_resolve(self):
        if self.value is not None:
            return '{}:[{} TO {}]'.format(
                self.search_field.field_name, *self.value
            )


class BackendQueryTypeRangeExclusive(BackendQueryTypeWhoosh):
    query_type = QueryTypeRangeExclusive

    def _do_resolve(self):
        if self.value is not None:
            return '{}:{{{} TO {}}}'.format(
                self.search_field.field_name, *self.value
            )


class BackendQueryTypeRegularExpression(BackendQueryTypeWhoosh):
    query_type = QueryTypeRegularExpression

    def _do_resolve(self):
        self.extra_kwargs['parser'].add_plugin(
            RegexPlugin()
        )

        if self.value is not None:
            return '{}:r"{}"'.format(
                self.search_field.field_name, self.value
            )


BackendQueryType.register(
    klass=BackendQueryTypeExact, search_backend=WhooshSearchBackend
)
BackendQueryType.register(
    klass=BackendQueryTypeFuzzy, search_backend=WhooshSearchBackend
)
BackendQueryType.register(
    klass=BackendQueryTypePartial, search_backend=WhooshSearchBackend
)
BackendQueryType.register(
    klass=BackendQueryTypeGreaterThan, search_backend=WhooshSearchBackend
)
BackendQueryType.register(
    klass=BackendQueryTypeGreaterThanOrEqual,
    search_backend=WhooshSearchBackend
)
BackendQueryType.register(
    klass=BackendQueryTypeLessThan, search_backend=WhooshSearchBackend
)
BackendQueryType.register(
    klass=BackendQueryTypeLessThanOrEqual, search_backend=WhooshSearchBackend
)
BackendQueryType.register(
    klass=BackendQueryTypeRange, search_backend=WhooshSearchBackend
)
BackendQueryType.register(
    klass=BackendQueryTypeRangeExclusive, search_backend=WhooshSearchBackend
)
BackendQueryType.register(
    klass=BackendQueryTypeRegularExpression,
    search_backend=WhooshSearchBackend
)
