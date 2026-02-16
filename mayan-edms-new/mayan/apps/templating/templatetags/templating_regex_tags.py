import re

from django.template import Library

from ..utils import process_regex_flags

register = Library()


@register.simple_tag(name='regex_findall')
def tag_regex_findall(pattern, string, **kwargs):
    """
    Return all non-overlapping matches of pattern in string, as a list of
    strings. {% regex_findall pattern string flags %}
    """
    flags = process_regex_flags(**kwargs)
    return re.findall(flags=flags, pattern=pattern, string=string)


@register.simple_tag(name='regex_match')
def tag_regex_match(pattern, string, **kwargs):
    """
    If zero or more characters at the beginning of string match the regular
    expression pattern, return a corresponding match object.
    {% regex_match pattern string flags %}
    """
    flags = process_regex_flags(**kwargs)
    return re.match(flags=flags, pattern=pattern, string=string)


@register.simple_tag(name='regex_search')
def tag_regex_search(pattern, string, **kwargs):
    """
    Scan through string looking for the first location where the regular
    expression pattern produces a match, and return a corresponding
    match object. {% regex_search pattern string flags %}
    """
    flags = process_regex_flags(**kwargs)
    return re.search(flags=flags, pattern=pattern, string=string)


@register.simple_tag(name='regex_sub')
def tag_regex_sub(pattern, repl, string, count=0, **kwargs):
    """
    Replacing the leftmost non-overlapping occurrences of pattern in
    string with repl. {% regex_sub pattern repl string count=0 flags %}
    """
    flags = process_regex_flags(**kwargs)
    return re.sub(
        count=count, flags=flags, pattern=pattern, repl=repl, string=string
    )
