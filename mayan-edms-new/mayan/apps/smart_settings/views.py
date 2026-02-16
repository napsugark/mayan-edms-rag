from django.conf import settings
from django.contrib import messages
from django.http import Http404
from django.urls import reverse, reverse_lazy
from django.utils.translation import gettext_lazy as _

from mayan.apps.views.generics import (
    ConfirmView, FormView, MultipleObjectConfirmActionView,
    SingleObjectListView
)

from .forms import SettingForm
from .icons import (
    icon_setting_cluster_configuration_save,
    icon_setting_cluster_namespace_list, icon_setting_namespace_detail,
    icon_setting_edit, icon_setting_revert
)
from .literals import MESSAGE_LOCAL_STORAGE_DISABLED
from .permissions import permission_settings_edit, permission_settings_view
from .settings import setting_cluster


class SettingClusterConfigurationFileSave(ConfirmView):
    post_action_redirect = reverse_lazy(
        viewname='settings:setting_cluster_namespace_list'
    )
    view_icon = icon_setting_cluster_configuration_save
    view_permission = permission_settings_edit

    def dispatch(self, request, *args, **kwargs):
        if settings.COMMON_DISABLE_LOCAL_STORAGE:
            messages.warning(
                message=MESSAGE_LOCAL_STORAGE_DISABLED, request=self.request
            )

        return super().dispatch(request=request, *args, **kwargs)

    def get_extra_context(self):
        return {
            'message': _(
                message='This will overwrite the content of the '
                'configuration file.'
            ),
            'title': _(message='Save settings to the configuration file?')
        }

    def view_action(self, form=None):
        setting_cluster.do_configuration_file_save()

        messages.success(
            message=_(
                message='Settings saved to configuration file successfully.'
            ), request=self.request
        )


class SettingClusterNamespaceListView(SingleObjectListView):
    extra_context = {
        'hide_link': True,
        'title': _(message='Setting namespaces')
    }
    view_icon = icon_setting_cluster_namespace_list
    view_permission = permission_settings_view

    def get_source_queryset(self):
        return setting_cluster.get_namespace_list()


class SettingValueEditView(FormView):
    form_class = SettingForm
    view_icon = icon_setting_edit
    view_permission = permission_settings_edit

    def dispatch(self, request, *args, **kwargs):
        if settings.COMMON_DISABLE_LOCAL_STORAGE:
            messages.warning(
                message=MESSAGE_LOCAL_STORAGE_DISABLED, request=self.request
            )

        self.object = self.get_object()

        return super().dispatch(request=request, *args, **kwargs)

    def form_valid(self, form):
        self.object.do_value_set(
            value=form.cleaned_data['value']
        )
        messages.success(
            message=_(message='Setting updated successfully.'),
            request=self.request
        )
        return super().form_valid(form=form)

    def get_extra_context(self):
        if self.object.get_is_overridden():
            messages.warning(
                message=_(
                    message='This setting is overridden by an environment '
                    'variable, changing its value will have no effect.'
                ), request=self.request
            )

        return {
            'hide_link': True,
            'navigation_object_list': ('object', 'setting_namespace'),
            'object': self.object,
            'setting_namespace': self.object.namespace,
            'title': _(message='Edit setting: %s') % self.object
        }

    def get_initial(self):
        return {'setting': self.object}

    def get_object(self):
        return setting_cluster.get_setting(
            global_name=self.kwargs['setting_global_name']
        )

    def get_post_action_redirect(self):
        return reverse(
            kwargs={'namespace_name': self.object.namespace.name},
            viewname='settings:setting_namespace_detail'
        )


class SettingValueRevertView(MultipleObjectConfirmActionView):
    pk_url_kwarg = 'setting_global_name'
    success_message_plural = _(
        message='%(count)d settings value reverted.'
    )
    success_message_single = _(
        message='Value of setting "%(object)s" reverted.'
    )
    success_message_singular = _(message='%(count)d setting value reverted.')
    title_plural = _(
        message='Revert the value of the %(count)d selected settings.'
    )
    title_single = _(message='Revert the value of setting: %(object)s')
    title_singular = _(
        message='Revert the value of %(count)d selected setting.'
    )
    view_icon = icon_setting_revert
    view_permission = permission_settings_edit

    def get_extra_context(self):
        object_list = self.object_list

        result = {
            'message': _(
                message='Unsaved changes will be lost.'
            )
        }

        if len(object_list) == 1:
            result['object'] = object_list[0]

        return result

    def get_object_first(self):
        return self.get_object_list()[0]

    def get_object_list(self):
        self.view_mode_single = True
        self.view_mode_multiple = False

        setting = setting_cluster.get_setting(
            global_name=self.kwargs['setting_global_name']
        )

        return (setting,)

    def object_action(self, form, instance):
        instance.do_value_revert()


class SettingNamespaceDetailView(SingleObjectListView):
    view_icon = icon_setting_namespace_detail
    view_permission = permission_settings_view

    def get_extra_context(self):
        namespace = self.get_namespace()

        return {
            'hide_object': True,
            'object': namespace,
            'subtitle': _(
                message='Settings inherited from an environment variable '
                'take precedence and cannot be changed in this view. '
            ),
            'title': _(message='Settings in namespace: %s') % namespace
        }

    def get_namespace(self):
        try:
            return setting_cluster.get_namespace(
                name=self.kwargs['namespace_name']
            )
        except KeyError:
            raise Http404(
                _(message='Namespace: %s, not found') % self.kwargs[
                    'namespace_name'
                ]
            )

    def get_source_queryset(self):
        setting_namespace = self.get_namespace()
        return setting_namespace.get_setting_list()
