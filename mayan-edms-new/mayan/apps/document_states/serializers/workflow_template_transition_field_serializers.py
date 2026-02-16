from django.utils.translation import gettext_lazy as _

from mayan.apps.rest_api import serializers
from mayan.apps.rest_api.relations import MultiKwargHyperlinkedIdentityField

from ..models.workflow_transition_field_models import WorkflowTransitionField


class WorkflowTransitionFieldSerializer(
    serializers.HyperlinkedModelSerializer
):
    url = MultiKwargHyperlinkedIdentityField(
        label=_(message='URL'), view_kwargs=(
            {
                'lookup_field': 'transition.workflow_id',
                'lookup_url_kwarg': 'workflow_template_id'
            },
            {
                'lookup_field': 'transition_id',
                'lookup_url_kwarg': 'workflow_template_transition_id'
            },
            {
                'lookup_field': 'pk',
                'lookup_url_kwarg': 'workflow_template_transition_field_id'
            }
        ), view_name='rest_api:workflow-template-transition-field-detail'
    )
    workflow_template_url = MultiKwargHyperlinkedIdentityField(
        label=_(message='Workflow template URL'), view_kwargs=(
            {
                'lookup_field': 'transition__workflow_id',
                'lookup_url_kwarg': 'workflow_template_id'
            },
        ), view_name='rest_api:workflow-template-detail'
    )
    workflow_transition_id = serializers.IntegerField(
        label=_(message='Workflow transition ID'), read_only=True,
        source='transition_id'
    )
    workflow_transition_url = MultiKwargHyperlinkedIdentityField(
        label=_(message='Workflow transition URL'), view_kwargs=(
            {
                'lookup_field': 'transition.workflow_id',
                'lookup_url_kwarg': 'workflow_template_id'
            },
            {
                'lookup_field': 'transition_id',
                'lookup_url_kwarg': 'workflow_template_transition_id'
            }
        ), view_name='rest_api:workflow-template-transition-detail'
    )

    class Meta:
        fields = (
            'field_type', 'name', 'help_text', 'id', 'label', 'lookup',
            'required', 'url', 'widget', 'widget_kwargs',
            'workflow_template_url', 'workflow_transition_id',
            'workflow_transition_url'
        )
        model = WorkflowTransitionField
        read_only_fields = (
            'id', 'url', 'workflow_template_url', 'workflow_transition_id',
            'workflow_transition_url'
        )
