from django.contrib.contenttypes.models import ContentType
from django.shortcuts import get_object_or_404

from mayan.apps.permissions.models import StoredPermission
from mayan.apps.permissions.serializers import PermissionSerializer
from mayan.apps.rest_api import generics
from mayan.apps.rest_api.api_view_mixins import (
    ExternalContentTypeObjectAPIViewMixin
)

from .classes import ModelPermission
from .permissions import permission_acl_edit, permission_acl_view
from .serializers import (
    ACLPermissionAddSerializer, ACLPermissionRemoveSerializer, ACLSerializer
)


class APIACLListView(
    ExternalContentTypeObjectAPIViewMixin, generics.ListCreateAPIView
):
    """
    get: Returns a list of all the object's access control lists
    post: Create a new access control list for the selected object.
    """
    mayan_external_object_permission_map = {
        'GET': permission_acl_view,
        'POST': permission_acl_edit
    }
    serializer_class = ACLSerializer

    def get_instance_extra_data(self):
        return {
            '_event_actor': self.request.user,
            'content_object': self.get_external_object()
        }

    def get_source_queryset(self):
        return self.get_external_object().acls.all()


class APIACLDetailView(
    ExternalContentTypeObjectAPIViewMixin, generics.RetrieveDestroyAPIView
):
    """
    delete: Delete the selected access control list.
    get: Returns the details of the selected access control list.
    """
    lookup_url_kwarg = 'acl_id'
    mayan_external_object_permission_map = {
        'DELETE': permission_acl_edit,
        'GET': permission_acl_view
    }
    serializer_class = ACLSerializer

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}

    def get_source_queryset(self):
        return self.get_external_object().acls.all()


class APIACLPermissionAddView(
    ExternalContentTypeObjectAPIViewMixin, generics.ObjectActionAPIView
):
    """
    post: Add a permission to an ACL.
    """
    lookup_url_kwarg = 'acl_id'
    mayan_external_object_permission_map = {'POST': permission_acl_edit}
    serializer_class = ACLPermissionAddSerializer

    def get_source_queryset(self):
        return self.get_external_object().acls

    def object_action(self, obj, request, serializer):
        obj.permissions_add(
            queryset=StoredPermission.objects.filter(
                pk=serializer.validated_data['permission'].stored_permission.pk
            ), user=self.request.user
        )


class APIACLPermissionListView(
    ExternalContentTypeObjectAPIViewMixin, generics.ListCreateAPIView
):
    """
    get: Returns the access control list permission list.
    post: Add a new permission to the selected access control list.
    """
    mayan_external_object_permission_map = {
        'GET': permission_acl_view,
        'POST': permission_acl_edit
    }
    serializer_class = PermissionSerializer

    def get_acl(self):
        return get_object_or_404(
            klass=self.get_external_object().acls, pk=self.kwargs['acl_id']
        )

    def get_source_queryset(self):
        return self.get_acl().permissions.all()


class APIACLPermissionRemoveView(
    ExternalContentTypeObjectAPIViewMixin, generics.ObjectActionAPIView
):
    """
    post: Remove a permission from an ACL.
    """
    lookup_url_kwarg = 'acl_id'
    mayan_external_object_permission_map = {'POST': permission_acl_edit}
    serializer_class = ACLPermissionRemoveSerializer

    def get_source_queryset(self):
        return self.get_external_object().acls

    def object_action(self, obj, request, serializer):
        obj.permissions_remove(
            queryset=StoredPermission.objects.filter(
                pk=serializer.validated_data['permission'].stored_permission.pk
            ), user=self.request.user
        )


class APIClassPermissionList(generics.ListAPIView):
    """
    Returns a list of all the available permissions for a class.
    """
    serializer_class = PermissionSerializer

    def get_content_type(self):
        return get_object_or_404(
            klass=ContentType, app_label=self.kwargs['app_label'],
            model=self.kwargs['model_name']
        )

    def get_source_queryset(self):
        return ModelPermission.get_for_class(
            klass=self.get_content_type().model_class()
        )
