from django.urls import reverse
from django.utils.translation import gettext_lazy as _

from mayan.apps.user_management.permissions import permission_user_view
from mayan.apps.user_management.querysets import get_user_queryset
from mayan.apps.user_management.views.view_mixins import (
    DynamicExternalUserViewMixin
)

from .generics import MultipleObjectDeleteView, SingleObjectListView
from .icons import (
    icon_user_confirm_property_view, icon_user_confirm_property_view_delete,
    icon_user_view_modes
)
from .models import UserConfirmView
from .view_mixins import ExternalObjectViewMixin


class UserConfirmViewDeleteView(MultipleObjectDeleteView):
    error_message = _(
        message='Error deleting confirm view "%(instance)s"; %(exception)s'
    )
    model = UserConfirmView
    pk_url_kwarg = 'user_confirmation_view_id'
    success_message_plural = _(
        message='%(count)d confirm view properties deleted successfully.'
    )
    success_message_single = _(
        message='Confirm view property "%(object)s" deleted successfully.'
    )
    success_message_singular = _(
        message='%(count)d confirm view property deleted successfully.'
    )
    title_plural = _(
        message='Delete the %(count)d selected confirm view property'
    )
    title_single = _(message='Delete confirm view property: %(object)s')
    title_singular = _(
        message='Delete the %(count)d selected confirm view property'
    )
    view_icon = icon_user_confirm_property_view_delete

    def get_post_action_redirect(self):
        return reverse(
            kwargs={'user_id': self.request.user.pk},
            viewname='views:user_views_confirm_list'
        )


class UserConfirmViewListView(
    DynamicExternalUserViewMixin, ExternalObjectViewMixin,
    SingleObjectListView
):
    external_object_permission = permission_user_view
    external_object_pk_url_kwarg = 'user_id'
    view_icon = icon_user_confirm_property_view

    def get_external_object_queryset(self):
        return get_user_queryset(user=self.request.user)

    def get_extra_context(self):
        return {
            'hide_link': True,
            'hide_object': True,
            'no_results_icon': icon_user_confirm_property_view,
            'no_results_text': _(
                message='Persistent confirmation properties modes remember '
                'the user preferences for a given confirmation view.'
            ),
            'no_results_title': _(
                message='No confirmation view properties available'
            ),
            'object': self.external_object,
            'title': _(
                message='Confirmation view properties for user: %s'
            ) % self.external_object
        }

    def get_source_queryset(self):
        return self.external_object.view_confirms.all()


class UserViewModeView(
    DynamicExternalUserViewMixin, ExternalObjectViewMixin,
    SingleObjectListView
):
    external_object_permission = permission_user_view
    external_object_pk_url_kwarg = 'user_id'
    view_icon = icon_user_view_modes

    def get_external_object_queryset(self):
        return get_user_queryset(user=self.request.user)

    def get_extra_context(self):
        return {
            'hide_link': True,
            'hide_object': True,
            'no_results_icon': icon_user_view_modes,
            'no_results_text': _(
                message='View modes control the format used to display a '
                'collection of objects.'
            ),
            'no_results_title': _(message='No view modes available'),
            'object': self.external_object,
            'title': _(
                message='Persistent view modes for user: %s'
            ) % self.external_object
        }

    def get_source_queryset(self):
        return self.external_object.view_modes.all()
