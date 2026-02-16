from django.apps import apps
from django.conf import settings
from django.template import Library
from django.template.exceptions import TemplateDoesNotExist
from django.template.loader import get_template
from django.utils.safestring import mark_safe
from django.utils.translation import gettext_lazy as _

from ..literals import COMMENT_APP_TEMPLATE_CACHE_DISABLE

app_templates_cache = {}
register = Library()


@register.filter(name='appearance_form_get_visile_fields_map')
def filter_appearance_form_get_visile_fields_map(form):
    field_map = {
        field.name: field for field in form.visible_fields()
    }
    return field_map


@register.filter(name='appearance_get_choice_value')
def filter_appearance_get_choice_value(field):
    try:
        return dict(field.field.choices)[
            field.value()
        ]
    except TypeError:
        return ', '.join(
            [
                subwidget.data['label'] for subwidget in field.subwidgets if subwidget.data['selected']
            ]
        )
    except KeyError:
        return _(message='None')


@register.filter(name='appearance_get_form_media_js')
def filter_appearance_get_form_media_js(form=None):
    if form:
        return [
            form.media.absolute_path(path) for path in form.media._js
        ]


@register.filter(name='appearance_object_list_count')
def filter_appearance_object_list_count(object_list):
    try:
        return object_list.count()
    except TypeError:
        return len(object_list)


@register.simple_tag(name='appearance_app_templates', takes_context=True)
def tag_appearance_app_templates(context, template_name):
    """
    Fetch the app templates for the requested `template_name`, render it with
    the current `request` from the `context`, and cache it for future use
    unless the template has the no caching comment.
    """
    result = []

    for app in apps.get_app_configs():
        template_id = '{}.{}'.format(app.label, template_name)
        if template_id not in app_templates_cache or settings.DEBUG:
            try:
                app_template = get_template(
                    '{}/app/{}.html'.format(app.label, template_name)
                )
                app_template_output = app_template.render(
                    context=context.flatten(), request=context.get('request')
                )

                if COMMENT_APP_TEMPLATE_CACHE_DISABLE not in app_template.template.source:
                    app_templates_cache[template_id] = app_template_output
            except TemplateDoesNotExist:
                """
                Non fatal just means that the app did not defined an app
                template of this name and purpose.
                """
                app_templates_cache[template_id] = ''
                app_template_output = ''
        else:
            app_template_output = app_templates_cache[template_id]

        result.append(app_template_output)

    return mark_safe(
        s=' '.join(result)
    )
