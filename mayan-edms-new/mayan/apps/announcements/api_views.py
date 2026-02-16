from mayan.apps.rest_api import generics

from .models import Announcement
from .permissions import (
    permission_announcement_create, permission_announcement_delete,
    permission_announcement_edit, permission_announcement_view
)
from .serializers import AnnouncementSerializer


class APIAnnouncementListView(generics.ListCreateAPIView):
    """
    get: Returns a list of all the announcements.
    post: Create a new announcement.
    """
    mayan_object_permission_map = {'GET': permission_announcement_view}
    mayan_view_permission_map = {'POST': permission_announcement_create}
    serializer_class = AnnouncementSerializer
    source_queryset = Announcement.objects.all()

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}


class APIAnnouncementView(generics.RetrieveUpdateDestroyAPIView):
    """
    delete: Delete the selected announcement.
    get: Return the details of the selected announcement.
    patch: Edit the selected announcement.
    put: Edit the selected announcement.
    """
    lookup_url_kwarg = 'announcement_id'
    mayan_object_permission_map = {
        'DELETE': permission_announcement_delete,
        'GET': permission_announcement_view,
        'PATCH': permission_announcement_edit,
        'PUT': permission_announcement_edit
    }
    serializer_class = AnnouncementSerializer
    source_queryset = Announcement.objects.all()

    def get_instance_extra_data(self):
        return {'_event_actor': self.request.user}
