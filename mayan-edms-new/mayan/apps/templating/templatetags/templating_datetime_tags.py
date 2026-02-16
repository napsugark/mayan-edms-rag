from datetime import timedelta

from dateutil.parser import ParserError, isoparse, parse

from django.template import Library, TemplateSyntaxError

register = Library()


@register.filter(name='date_parse_iso')
def filter_date_parse_iso(date_string):
    """
    Takes an ISO-8601 string and converts it into a datetime object.
    """

    try:
        result = isoparse(str_in=date_string)
    except ValueError as exception:
        raise TemplateSyntaxError(
            str(exception)
        ) from exception
    else:
        return result


@register.simple_tag(name='date_parse')
def tag_date_parse(
    date_string, dayfirst=None, fuzzy=None, fuzzy_with_tokens=None,
    ignoretz=None, yearfirst=None
):
    """
    Takes a string and converts it into a datetime object.
    """

    try:
        result = parse(
            dayfirst=dayfirst, fuzzy=fuzzy,
            fuzzy_with_tokens=fuzzy_with_tokens, ignoretz=ignoretz,
            timestr=date_string, yearfirst=yearfirst
        )
    except ParserError as exception:
        raise TemplateSyntaxError(
            str(exception)
        ) from exception
    else:
        return result


@register.simple_tag(name='timedelta')
def tag_timedelta(date, **kwargs):
    """
    Takes a datetime object and applies a timedelta.
    """

    return date + timedelta(**kwargs)
