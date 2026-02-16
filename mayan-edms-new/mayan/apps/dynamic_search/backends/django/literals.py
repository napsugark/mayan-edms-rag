from django.db import models

from ...search_query_types import (
    QueryTypeExact, QueryTypeFuzzy, QueryTypeGreaterThan,
    QueryTypeGreaterThanOrEqual, QueryTypeLessThan, QueryTypeLessThanOrEqual,
    QueryTypePartial, QueryTypeRange, QueryTypeRangeExclusive,
    QueryTypeRegularExpression
)
from ...value_transformations import (
    ValueTransformationHyphenReplace, ValueTransformationHyphenStrip,
    ValueTransformationToBoolean, ValueTransformationToDateTime,
    ValueTransformationToInteger, ValueTransformationToString
)

DEFAULT_FUZZY_SLOP = 2

DJANGO_TO_DJANGO_FIELD_MAP = {
    models.AutoField: {
        'field': models.AutoField,
        'query_type_list': [
            QueryTypeExact, QueryTypePartial, QueryTypeGreaterThan,
            QueryTypeGreaterThanOrEqual, QueryTypeLessThan,
            QueryTypeLessThanOrEqual, QueryTypeRange, QueryTypeRangeExclusive
        ],
        'transformations': {
            'search': [ValueTransformationToInteger]
        }
    },
    models.BigIntegerField: {
        'field': models.BigIntegerField,
        'query_type_list': [
            QueryTypeExact, QueryTypeGreaterThan,
            QueryTypeGreaterThanOrEqual, QueryTypeLessThan,
            QueryTypeLessThanOrEqual, QueryTypeRange, QueryTypeRangeExclusive
        ],
        'transformations': {
            'search': [ValueTransformationToInteger]
        }
    },
    models.BooleanField: {
        'field': models.BooleanField,
        'query_type_list': [
            QueryTypeExact
        ],
        'transformations': {
            'search': [
                ValueTransformationToString, ValueTransformationToBoolean
            ]
        }
    },
    models.CharField: {
        'field': models.CharField,
        'query_type_list': [
            QueryTypeExact, QueryTypeFuzzy, QueryTypePartial,
            QueryTypeRegularExpression, QueryTypeGreaterThan,
            QueryTypeGreaterThanOrEqual, QueryTypeLessThan,
            QueryTypeLessThanOrEqual
        ],
        'transformations': {
            'search': [
                ValueTransformationToString, ValueTransformationHyphenReplace
            ]
        }
    },
    models.DateTimeField: {
        'field': models.DateTimeField,
        'query_type_list': [
            QueryTypeExact, QueryTypeGreaterThan,
            QueryTypeGreaterThanOrEqual, QueryTypeLessThan,
            QueryTypeLessThanOrEqual, QueryTypeRange, QueryTypeRangeExclusive
        ],
        'transformations': {
            'search': [ValueTransformationToDateTime]
        }
    },
    models.EmailField: {
        'field': models.EmailField,
        'query_type_list': [
            QueryTypeExact, QueryTypePartial
        ],
        'transformations': {
            'search': [ValueTransformationToString]
        }
    },
    models.IntegerField: {
        'field': models.IntegerField,
        'query_type_list': [
            QueryTypeExact, QueryTypeGreaterThan,
            QueryTypeGreaterThanOrEqual, QueryTypeLessThan,
            QueryTypeLessThanOrEqual, QueryTypeRange, QueryTypeRangeExclusive
        ],
        'transformations': {
            'search': [ValueTransformationToInteger]
        }
    },
    models.PositiveBigIntegerField: {
        'field': models.PositiveBigIntegerField,
        'query_type_list': [
            QueryTypeExact, QueryTypeGreaterThan,
            QueryTypeGreaterThanOrEqual, QueryTypeLessThan,
            QueryTypeLessThanOrEqual, QueryTypeRange, QueryTypeRangeExclusive
        ],
        'transformations': {
            'search': [ValueTransformationToInteger]
        }
    },
    models.PositiveIntegerField: {
        'field': models.PositiveIntegerField,
        'query_type_list': [
            QueryTypeExact, QueryTypeGreaterThan,
            QueryTypeGreaterThanOrEqual, QueryTypeLessThan,
            QueryTypeLessThanOrEqual, QueryTypeRange, QueryTypeRangeExclusive
        ],
        'transformations': {
            'search': [ValueTransformationToInteger]
        }
    },
    models.TextField: {
        'field': models.TextField,
        'query_type_list': [
            QueryTypeExact, QueryTypeFuzzy, QueryTypePartial,
            QueryTypeRegularExpression, QueryTypeGreaterThan,
            QueryTypeGreaterThanOrEqual, QueryTypeLessThan,
            QueryTypeLessThanOrEqual
        ],
        'transformations': {
            'search': [
                ValueTransformationToString, ValueTransformationHyphenReplace
            ]
        }
    },
    models.UUIDField: {
        'field': models.UUIDField,
        'query_type_list': [
            QueryTypeExact, QueryTypePartial, QueryTypeRegularExpression
        ],
        'transformations': {
            'search': [
                ValueTransformationToString, ValueTransformationHyphenStrip
            ]
        }
    }
}

MAXIMUM_FUZZY_OPTIONS = 50
