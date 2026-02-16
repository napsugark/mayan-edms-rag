from django.utils.translation import gettext_lazy as _

from rest_framework.reverse import reverse

from mayan.apps.rest_api import serializers
from mayan.apps.rest_api.relations import FilteredPrimaryKeyRelatedField

from ..models.workflow_transition_models import WorkflowTransition

from .workflow_template_state_serializers import (
    WorkflowTemplateStateSerializer
)


class WorkflowTemplateTransitionSerializer(
    serializers.HyperlinkedModelSerializer
):
    destination_state = WorkflowTemplateStateSerializer(
        label=_(message='Destination state'), read_only=True
    )
    destination_state_id = FilteredPrimaryKeyRelatedField(
        help_text=_(
            message='Primary key of the destination state to be added.'
        ), label=_(message='Destination state ID'),
        source_queryset_method='get_workflow_template_state_queryset',
        write_only=True
    )
    field_list_url = serializers.SerializerMethodField(
        label=_(message='Field list URL')
    )
    origin_state = WorkflowTemplateStateSerializer(
        label=_(message='Origin state'), read_only=True
    )
    origin_state_id = FilteredPrimaryKeyRelatedField(
        help_text=_(
            message='Primary key of the origin state to be added.'
        ), label=_(message='Origin state ID'),
        source_queryset_method='get_workflow_template_state_queryset',
        write_only=True
    )
    trigger_list_url = serializers.SerializerMethodField(
        label=_(message='Trigger list URL')
    )
    url = serializers.SerializerMethodField(
        label=_(message='URL')
    )
    workflow_template_id = serializers.IntegerField(
        label=_(message='Workflow template ID'), read_only=True,
        source='workflow_id'
    )
    workflow_template_url = serializers.SerializerMethodField(
        label=_(message='Workflow template URL')
    )

    class Meta:
        fields = (
            'condition', 'destination_state', 'destination_state_id',
            'field_list_url', 'id', 'label', 'origin_state',
            'origin_state_id', 'trigger_list_url', 'url',
            'workflow_template_id', 'workflow_template_url'
        )
        model = WorkflowTransition
        read_only_fields = (
            'field_list_url', 'id', 'trigger_list_url', 'url',
            'workflow_template_id', 'workflow_template_url'
        )

    def create(self, validated_data):
        validated_data['destination_state'] = validated_data.pop(
            'destination_state_id'
        )
        validated_data['origin_state'] = validated_data.pop(
            'origin_state_id'
        )

        return super().create(
            validated_data=validated_data
        )

    def get_field_list_url(self, instance):
        return reverse(
            format=self.context['format'], kwargs={
                'workflow_template_id': instance.workflow_id,
                'workflow_template_transition_id': instance.pk
            }, request=self.context['request'],
            viewname='rest_api:workflow-template-transition-field-list'
        )

    def get_trigger_list_url(self, instance):
        return reverse(
            format=self.context['format'], kwargs={
                'workflow_template_id': instance.workflow_id,
                'workflow_template_transition_id': instance.pk
            }, request=self.context['request'],
            viewname='rest_api:workflow-template-transition-trigger-list'
        )

    def get_url(self, instance):
        return reverse(
            format=self.context['format'], kwargs={
                'workflow_template_id': instance.workflow.pk,
                'workflow_template_transition_id': instance.pk
            }, request=self.context['request'],
            viewname='rest_api:workflow-template-transition-detail'
        )

    def get_workflow_template_state_queryset(self):
        return self.context['workflow_template'].states.all()

    def get_workflow_template_url(self, instance):
        return reverse(
            format=self.context['format'], kwargs={
                'workflow_template_id': instance.workflow.pk
            }, request=self.context['request'],
            viewname='rest_api:workflow-template-detail'
        )

    def update(self, instance, validated_data):
        validated_data['destination_state'] = validated_data.pop(
            'destination_state_id'
        )
        validated_data['origin_state'] = validated_data.pop(
            'origin_state_id'
        )

        return super().update(
            instance=instance, validated_data=validated_data
        )
