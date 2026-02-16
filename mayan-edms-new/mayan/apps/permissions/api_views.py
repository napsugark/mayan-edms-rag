from django.contrib.auth.models import Group

from mayan.apps.rest_api import generics
from mayan.apps.rest_api.api_view_mixins import ExternalObjectAPIViewMixin
from mayan.apps.user_management.permissions import permission_group_view
from mayan.apps.user_management.serializers import GroupSerializer

from .classes import Permission
from .models import Role, StoredPermission
from .permissions import (
    permission_role_create, permission_role_delete, permission_role_edit,
    permission_role_view
)
from .serializers import (
    PermissionSerializer, RoleGroupAddSerializer, RoleGroupRemoveSerializer,
    RolePermissionAddSerializer, RolePermissionRemoveSerializer,
    RoleSerializer
)


class APIPermissionList(generics.ListAPIView):
    """
    get: Returns a list of all the available permissions.
    """
    serializer_class = PermissionSerializer
    source_queryset = Permission.all()


class APIRoleListView(generics.ListCreateAPIView):
    """
    get: Returns a list of all the roles.
    post: Create a new role.
    """
    mayan_object_permission_map = {'GET': permission_role_view}
    mayan_view_permission_map = {'POST': permission_role_create}
    serializer_class = RoleSerializer
    source_queryset = Role.objects.all()

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}


class APIRoleDetailView(generics.RetrieveUpdateDestroyAPIView):
    """
    delete: Delete the selected role.
    get: Return the details of the selected role.
    patch: Edit the selected role.
    put: Edit the selected role.
    """
    lookup_url_kwarg = 'role_id'
    mayan_object_permission_map = {
        'GET': permission_role_view,
        'PUT': permission_role_edit,
        'PATCH': permission_role_edit,
        'DELETE': permission_role_delete
    }
    serializer_class = RoleSerializer
    source_queryset = Role.objects.all()

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}


class APIRoleGroupListView(
    ExternalObjectAPIViewMixin, generics.ListAPIView
):
    """
    get: Returns a list of all the groups granted to a particular role.
    """
    external_object_class = Role
    external_object_pk_url_kwarg = 'role_id'
    mayan_external_object_permission_map = {'GET': permission_role_view}
    mayan_object_permission_map = {'GET': permission_group_view}
    serializer_class = GroupSerializer

    def get_source_queryset(self):
        return self.get_external_object().groups.all()


class APIRoleGroupAddView(generics.ObjectActionAPIView):
    """
    post: Add a group to a role.
    """
    lookup_url_kwarg = 'role_id'
    mayan_object_permission_map = {'POST': permission_role_edit}
    serializer_class = RoleGroupAddSerializer
    source_queryset = Role.objects.all()

    def object_action(self, obj, request, serializer):
        obj.groups_add(
            queryset=Group.objects.filter(
                pk=serializer.validated_data['group'].pk
            ), user=self.request.user
        )


class APIRoleGroupRemoveView(generics.ObjectActionAPIView):
    """
    post: Remove a group from a role.
    """
    lookup_url_kwarg = 'role_id'
    mayan_object_permission_map = {'POST': permission_role_edit}
    serializer_class = RoleGroupRemoveSerializer
    source_queryset = Role.objects.all()

    def object_action(self, obj, request, serializer):
        obj.groups_remove(
            queryset=Group.objects.filter(
                pk=serializer.validated_data['group'].pk
            ), user=self.request.user
        )


class APIRolePermissionListView(
    ExternalObjectAPIViewMixin, generics.ListAPIView
):
    """
    get: Returns a list of all the permissions granted to a particular role.
    """
    external_object_class = Role
    external_object_pk_url_kwarg = 'role_id'
    mayan_external_object_permission_map = {'GET': permission_role_view}
    serializer_class = PermissionSerializer

    def get_source_queryset(self):
        return self.get_external_object().permissions.all()


class APIRolePermissionAddView(generics.ObjectActionAPIView):
    """
    post: Add a permission to a role.
    """
    lookup_url_kwarg = 'role_id'
    mayan_object_permission_map = {'POST': permission_role_edit}
    serializer_class = RolePermissionAddSerializer
    source_queryset = Role.objects.all()

    def object_action(self, obj, request, serializer):
        obj.permissions_add(
            queryset=StoredPermission.objects.filter(
                pk=serializer.validated_data['permission'].stored_permission.pk
            ), user=self.request.user
        )


class APIRolePermissionRemoveView(generics.ObjectActionAPIView):
    """
    post: Remove a permission from a role.
    """
    lookup_url_kwarg = 'role_id'
    mayan_object_permission_map = {'POST': permission_role_edit}
    serializer_class = RolePermissionRemoveSerializer
    source_queryset = Role.objects.all()

    def object_action(self, obj, request, serializer):
        obj.permissions_remove(
            queryset=StoredPermission.objects.filter(
                pk=serializer.validated_data['permission'].stored_permission.pk
            ), user=self.request.user
        )
