from django.utils.translation import gettext_lazy as _

from mayan.apps.events.classes import EventTypeNamespace

namespace = EventTypeNamespace(
    label=_(message='Workflows'), name='document_states'
)

event_workflow_instance_created = namespace.add_event_type(
    label=_(message='Workflow instance created'), name='workflow_instance_created'
)
event_workflow_instance_transitioned = namespace.add_event_type(
    label=_(message='Workflow instance transitioned'),
    name='workflow_instance_transitioned'
)

event_workflow_template_created = namespace.add_event_type(
    label=_(message='Workflow created'), name='workflow_created'
)
event_workflow_template_edited = namespace.add_event_type(
    label=_(message='Workflow edited'), name='workflow_edited'
)
