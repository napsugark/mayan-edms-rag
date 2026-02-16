import logging

from furl import furl

from django.apps import apps
from django.core.exceptions import ImproperlyConfigured, PermissionDenied
from django.template import RequestContext, Variable, VariableDoesNotExist
from django.template.defaulttags import URLNode
from django.urls import resolve, reverse
from django.utils.encoding import force_str

from mayan.apps.common.settings import setting_home_view
from mayan.apps.permissions.classes import Permission

from .class_mixins import TemplateObjectMixin

logger = logging.getLogger(name=__name__)


class Link(TemplateObjectMixin):
    _registry = {}

    @staticmethod
    def conditional_active_by_view_name(context, resolved_link):
        return resolved_link.link.view == resolved_link.current_view_name

    @classmethod
    def get(cls, name):
        return cls._registry[name]

    @classmethod
    def remove(cls, name):
        del cls._registry[name]

    def __init__(
        self, text=None, view=None, args=None, badge_text=None,
        condition=None, conditional_active=None, conditional_disable=None,
        description=None, html_data=None, html_extra_classes=None, icon=None,
        keep_query=False, kwargs=None, name=None, permission=None, query=None,
        remove_from_query=None, tags=None, title=None, url=None
    ):
        self.args = args or []
        self.badge_text = badge_text
        self.condition = condition
        self.conditional_active = conditional_active or Link.conditional_active_by_view_name
        self.conditional_disable = conditional_disable
        self.description = description
        self.html_data = html_data
        self.html_data_resolved = None
        self.html_extra_classes = html_extra_classes
        self._icon = icon
        self.keep_query = keep_query
        self._kwargs = kwargs or {}
        self.name = name
        self._permission = permission
        self.query = query or {}
        self.remove_from_query = remove_from_query or []
        self.tags = tags
        self.text = text
        self.title = title
        self.view = view
        self.url = url

        if name:
            self.__class__._registry[name] = self

    def get_icon(self, context=None):
        return self._icon

    def get_kwargs(self, context):
        try:
            return self._kwargs(context)
        except TypeError:
            # Is not a callable.
            return self._kwargs

    def get_permission_object(self, context):
        return None

    def get_permission(self, context):
        return self._permission

    def get_resolved_object(self, context):
        try:
            return Variable(
                var='object'
            ).resolve(context=context)
        except VariableDoesNotExist:
            """No object variable in the context"""

    def resolve(self, context=None, request=None, resolved_object=None):
        AccessControlList = apps.get_model(
            app_label='acls', model_name='AccessControlList'
        )

        if not context and not request:
            raise ImproperlyConfigured(
                'Must provide a context or a request in order to resolve the '
                'link.'
            )

        if not context:
            context = RequestContext(request=request)

        request = self.get_request(context=context, request=request)

        current_path = request.META['PATH_INFO']
        current_view_name = resolve(path=current_path).view_name

        if not resolved_object:
            resolved_object = self.get_resolved_object(context=context)

        # If we were passed an instance of the view context object we are
        # resolving, inject it into the context. This helps resolve links for
        # object lists.
        if resolved_object:
            context['resolved_object'] = resolved_object

        # ACL is tested against the resolved_object, {{ object }}
        # or a custom object returned by the link subclass.
        permission_object = self.get_permission_object(context=context) or resolved_object

        # If this link has a required permission check that the user has it
        # too.
        permission = self.get_permission(context=context)

        if permission:
            if permission_object:
                try:
                    AccessControlList.objects.check_access(
                        obj=permission_object, permission=permission,
                        user=request.user
                    )
                except PermissionDenied:
                    return None
            else:
                try:
                    Permission.check_user_permission(
                        permission=permission, user=request.user
                    )
                except PermissionDenied:
                    return None

        # Check to see if link has conditional display function and only
        # display it if the result of the conditional display function is
        # True.
        if not self.check_condition(context=context, resolved_object=resolved_object):
            return None

        resolved_link = ResolvedLink(
            current_view_name=current_view_name, link=self
        )

        if self.view:
            view_name = Variable(
                var='"{}"'.format(self.view)
            )
            if isinstance(self.args, list) or isinstance(self.args, tuple):
                args = [
                    Variable(var=arg) for arg in self.args
                ]
            else:
                args = [
                    Variable(var=self.args)
                ]

            kwargs = self.get_kwargs(context=context)

            kwargs = {
                key: Variable(var=value) for key, value in kwargs.items()
            }

            # Use Django's exact {% url %} code to resolve the link.
            node = URLNode(
                view_name=view_name, args=args, kwargs=kwargs, asvar=None
            )
            try:
                resolved_link.url = node.render(context=context)
            except VariableDoesNotExist as exception:
                """Not critical, ignore"""
                logger.debug(
                    'VariableDoesNotExist when resolving link "%s" '
                    'URL; %s', self.text, exception
                )
            except Exception as exception:
                logger.error(
                    'Error resolving link "%s" URL; %s', self.text,
                    exception, exc_info=True
                )
        elif self.url is not None:
            resolved_link.url = self.url

        if self.html_data:
            result = {}
            for key, value in self.html_data.items():
                try:
                    resolved_value = Variable(
                        var=value
                    ).resolve(context=context)
                except VariableDoesNotExist:
                    """No object variable in the context"""
                    resolved_value = value

                result[key] = resolved_value

            self.html_data_resolved = result

        # This is for links that should be displayed but that are not
        # clickable.
        if self.conditional_disable:
            resolved_link.disabled = self.conditional_disable(
                context=context
            )
        else:
            resolved_link.disabled = False

        # Lets a new link keep the same URL query string of the current URL.
        if self.keep_query:
            # Sometimes we are required to remove a key from the URL query.
            parsed_url = furl(
                force_str(
                    s=request.get_full_path() or request.META.get(
                        'HTTP_REFERER', reverse(setting_home_view.value)
                    )
                )
            )

            for key in self.remove_from_query:
                try:
                    parsed_url.query.remove(key)
                except KeyError:
                    pass

            # Use the link's URL but with the previous URL querystring.
            new_url = furl(url=resolved_link.url)
            new_url.args = parsed_url.querystr
            resolved_link.url = new_url.url

        if self.query:
            new_url = furl(url=resolved_link.url)
            for key, value in self.query.items():
                try:
                    resolved_variable = Variable(
                        var=value
                    ).resolve(context=context)
                except VariableDoesNotExist:
                    """
                    Not fatal. Variable resolution here is perform as if
                    resolving a template variable. Non existing results
                    are not updated.
                    """
                else:
                    new_url.args[key] = resolved_variable

            resolved_link.url = new_url.url

        resolved_link.context = context

        return resolved_link


class ResolvedLink:
    def __init__(self, link, current_view_name):
        self.context = None
        self.current_view_name = current_view_name
        self.disabled = False
        self.link = link
        self.request = None
        self.url = '#'

    def __repr__(self):
        return '<ResolvedLink: {}>'.format(self.url)

    @property
    def active(self):
        conditional_active = self.link.conditional_active
        if conditional_active:
            return conditional_active(
                context=self.context, resolved_link=self
            )

    @property
    def badge_text(self):
        if self.link.badge_text:
            return self.link.badge_text(context=self.context)

    @property
    def description(self):
        return self.link.description

    def get_icon(self, context=None):
        return self.link.get_icon(context=context)

    @property
    def html_data(self):
        return self.link.html_data_resolved

    @property
    def html_extra_classes(self):
        return self.link.html_extra_classes or ''

    @property
    def tags(self):
        return self.link.tags

    @property
    def text(self):
        try:
            return self.link.text(context=self.context)
        except TypeError:
            return self.link.text

    @property
    def title(self):
        return self.link.title


class Separator(Link):
    """
    Menu separator. Renders to an <hr> tag.
    """
    def __init__(self, *args, **kwargs):
        self._icon = None
        self.text = None
        self.view = None

    def resolve(self, *args, **kwargs):
        result = ResolvedLink(current_view_name=None, link=self)
        result.separator = True
        return result


class Text(Link):
    """
    Menu text. Renders to a plain <li> tag.
    """
    def __init__(self, *args, **kwargs):
        self.html_extra_classes = kwargs.get('html_extra_classes', '')
        self._icon = None
        self.text = kwargs.get('text')
        self.view = None

    def resolve(self, *args, **kwargs):
        result = ResolvedLink(current_view_name=None, link=self)
        result.context = kwargs.get('context')
        result.text_span = True
        return result
