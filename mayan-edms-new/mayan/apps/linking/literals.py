from django.utils.translation import gettext_lazy as _

INCLUSION_AND = '&'
INCLUSION_OR = '|'

INCLUSION_CHOICES = (
    (INCLUSION_AND, _(message='and')),
    (INCLUSION_OR, _(message='or'))
)

OPERATOR_CHOICES = (
    ('exact', _(message='is equal to')),
    ('iexact', _(message='is equal to (case insensitive)')),
    ('contains', _(message='contains')),
    ('icontains', _(message='contains (case insensitive)')),
    ('in', _(message='is in')),
    ('gt', _(message='is greater than')),
    ('gte', _(message='is greater than or equal to')),
    ('lt', _(message='is less than')),
    ('lte', _(message='is less than or equal to')),
    ('startswith', _(message='starts with')),
    ('istartswith', _(message='starts with (case insensitive)')),
    ('endswith', _(message='ends with')),
    ('iendswith', _(message='ends with (case insensitive)')),
    ('regex', _(message='is in regular expression')),
    ('iregex', _(message='is in regular expression (case insensitive)'))
)
