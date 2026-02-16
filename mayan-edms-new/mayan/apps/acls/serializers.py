from django.utils.translation import gettext_lazy as _

from mayan.apps.common.serializers import ContentTypeSerializer
from mayan.apps.permissions.classes import Permission
from mayan.apps.permissions.serializers import RoleSerializer
from mayan.apps.rest_api import serializers
from mayan.apps.rest_api.relations import (
    FilteredPrimaryKeyRelatedField, MultiKwargHyperlinkedIdentityField
)

from .models import AccessControlList


class ACLSerializer(serializers.ModelSerializer):
    content_type = ContentTypeSerializer(
        label=_(message='Content type'), read_only=True
    )
    permissions_add_url = MultiKwargHyperlinkedIdentityField(
        label=_(message='Permissions add URL'), view_kwargs=(
            {
                'lookup_field': 'content_type.app_label',
                'lookup_url_kwarg': 'app_label'
            },
            {
                'lookup_field': 'content_type.model',
                'lookup_url_kwarg': 'model_name'
            },
            {
                'lookup_field': 'object_id',
                'lookup_url_kwarg': 'object_id'
            },
            {
                'lookup_field': 'pk',
                'lookup_url_kwarg': 'acl_id'
            }
        ), view_name='rest_api:accesscontrollist-permission-add'
    )
    permissions_remove_url = MultiKwargHyperlinkedIdentityField(
        label=_(message='Permissions remove URL'), view_kwargs=(
            {
                'lookup_field': 'content_type.app_label',
                'lookup_url_kwarg': 'app_label'
            },
            {
                'lookup_field': 'content_type.model',
                'lookup_url_kwarg': 'model_name'
            },
            {
                'lookup_field': 'object_id',
                'lookup_url_kwarg': 'object_id'
            },
            {
                'lookup_field': 'pk',
                'lookup_url_kwarg': 'acl_id'
            }
        ), view_name='rest_api:accesscontrollist-permission-remove'
    )
    permissions_url = MultiKwargHyperlinkedIdentityField(
        label=_(message='Permissions URL'), view_kwargs=(
            {
                'lookup_field': 'content_type.app_label',
                'lookup_url_kwarg': 'app_label'
            },
            {
                'lookup_field': 'content_type.model',
                'lookup_url_kwarg': 'model_name'
            },
            {
                'lookup_field': 'object_id',
                'lookup_url_kwarg': 'object_id'
            },
            {
                'lookup_field': 'pk',
                'lookup_url_kwarg': 'acl_id'
            }
        ), view_name='rest_api:accesscontrollist-permission-list'
    )
    role = RoleSerializer(
        label=_(message='Role'), read_only=True
    )
    role_id = serializers.IntegerField(
        label=_(message='Role ID'), write_only=True
    )
    url = MultiKwargHyperlinkedIdentityField(
        label=_(message='URL'), view_kwargs=(
            {
                'lookup_field': 'content_type.app_label',
                'lookup_url_kwarg': 'app_label'
            },
            {
                'lookup_field': 'content_type.model',
                'lookup_url_kwarg': 'model_name'
            },
            {
                'lookup_field': 'object_id',
                'lookup_url_kwarg': 'object_id'
            },
            {
                'lookup_field': 'pk',
                'lookup_url_kwarg': 'acl_id'
            }
        ), view_name='rest_api:accesscontrollist-detail'
    )

    class Meta:
        create_only_fields = ('role_id',)
        fields = (
            'content_type', 'id', 'object_id', 'permissions_add_url',
            'permissions_remove_url', 'permissions_url', 'role', 'role_id',
            'url'
        )
        model = AccessControlList
        read_only_fields = ('content_type', 'object_id')


class ACLPermissionAddSerializer(serializers.Serializer):
    permission = FilteredPrimaryKeyRelatedField(
        help_text=_(message='Primary key of the permission to add to the ACL.'),
        label=_(message='Permission ID'), source_queryset=Permission.all()
    )


class ACLPermissionRemoveSerializer(serializers.Serializer):
    permission = FilteredPrimaryKeyRelatedField(
        help_text=_(message='Primary key of the permission to remove from the ACL.'),
        label=_(message='Permission ID'), source_queryset=Permission.all()
    )
