from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='Download files'), name='download_files'
)

event_download_file_created = namespace.add_event_type(
    label=_(message='Download file created'), name='created'
)
event_download_file_deleted = namespace.add_event_type(
    label=_(message='Download file deleted'), name='deleted'
)
event_download_file_downloaded = namespace.add_event_type(
    label=_(message='Download file downloaded'), name='downloaded'
)
