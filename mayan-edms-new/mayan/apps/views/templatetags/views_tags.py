import logging

from django.template import Context, Library, Variable, VariableDoesNotExist
from django.template.defaultfilters import truncatechars
from django.template.loader import get_template
from django.utils.translation import gettext_lazy as _

from mayan.apps.appearance.settings import setting_max_title_length

from ..icons import icon_list_mode_items, icon_list_mode_list
from ..literals import TEXT_LIST_AS_ITEMS_PARAMETER
from ..settings import setting_paging_argument

logger = logging.getLogger(name=__name__)
register = Library()


@register.simple_tag(name='views_calculate_title', takes_context=True)
def tag_views_calculate_title(context):
    title = ''
    title_full = ''

    if context.get('title'):
        title_full = context.get('title')
        title = truncatechars(
            title_full, arg=setting_max_title_length.value
        )
    else:
        if context.get('delete_view'):
            title = _(message='Confirm delete')
            title_full = title
        else:
            if context.get('form'):
                if context.get('object'):
                    title = _(message='Edit %s') % context.get('object')
                    title_full = title
                else:
                    title = _(message='Confirm')
                    title_full = title
            else:
                if context.get('read_only'):
                    title = _(message='Details for: %s') % context.get('object')
                    title_full = title
                else:
                    if context.get('object'):
                        title = _(message='Edit: %s') % context.get('object')
                        title_full = title
                    else:
                        if context.get('create_view') or context.get('form'):
                            title = _(message='Create')
                            title_full = title

    return {'title': title, 'title_full': title_full}


@register.simple_tag(name='views_get_list_mode_icon', takes_context=True)
def tag_views_get_list_mode_icon(context):
    if context.get('list_as_items', False):
        return icon_list_mode_list
    else:
        return icon_list_mode_items


@register.simple_tag(
    name='views_get_list_mode_querystring', takes_context=True
)
def tag_views_get_list_mode_querystring(context):
    list_as_items = context.get('list_as_items', False)

    if list_as_items:
        list_mode = 'list'
    else:
        list_mode = 'items'

    kwargs = {
        TEXT_LIST_AS_ITEMS_PARAMETER: list_mode
    }

    return tag_views_update_query_string(context=context, **kwargs)


@register.simple_tag(name='views_get_paging_query_string', takes_context=True)
def tag_views_get_paging_query_string(context, page_number):
    kwargs = {
        setting_paging_argument.value: page_number
    }
    return tag_views_update_query_string(context=context, **kwargs)


@register.simple_tag(name='views_get_proper_elided_page_range')
def tag_views_get_proper_elided_page_range(
    paginator, number, on_each_side=None, on_ends=None
):
    kwargs = {
        'number': number
    }

    if on_each_side:
        kwargs['on_each_side'] = on_each_side

    if on_ends:
        kwargs['on_ends'] = on_ends

    return paginator.get_elided_page_range(**kwargs)


@register.simple_tag(name='views_render_subtemplate', takes_context=True)
def tag_views_render_subtemplate(context, template_name, template_context):
    """
    Renders the specified template with the mixed parent and
    subtemplate contexts.
    """
    new_context = Context(
        context.flatten()
    )
    new_context.update(
        Context(template_context)
    )
    return get_template(template_name=template_name).render(
        context=new_context.flatten()
    )


@register.simple_tag(name='views_update_query_string', takes_context=True)
def tag_views_update_query_string(context, **kwargs):
    try:
        request = context.request
    except AttributeError:
        # Simple request extraction failed. Might not be a view context.
        # Try alternate method.
        try:
            request = Variable(var='request').resolve(context=context)
        except VariableDoesNotExist:
            # There is no request variable, most probable a 500 in a test
            # view. Don't return any resolved request.
            logger.warning('No request variable, aborting request resolution')
            return ''

    # We do this to get an mutable copy we can modify.
    query = request.GET.copy()

    for key, value in kwargs.items():
        query[key] = value

    return '?{}'.format(
        query.urlencode()
    )
