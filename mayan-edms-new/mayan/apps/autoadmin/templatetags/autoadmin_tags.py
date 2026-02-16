from django.template import Library

from ..models import AutoAdminSingleton

register = Library()


@register.inclusion_tag(
    filename='autoadmin/credentials.html', name='autoadmin_partial'
)
def tag_autoadmin_partial():
    try:
        return {'autoadmin_properties': AutoAdminSingleton.objects.get()}
    except AutoAdminSingleton.DoesNotExist:
        return {'autoadmin_properties': None}


@register.simple_tag(name='autoadmin_properties', takes_context=True)
def tag_autoadmin_properties(context):
    try:
        context['autoadmin_properties'] = AutoAdminSingleton.objects.get()
    except AutoAdminSingleton.DoesNotExist:
        context['autoadmin_properties'] = None

    return ''
