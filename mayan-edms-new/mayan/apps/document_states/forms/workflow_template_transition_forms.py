from mayan.apps.forms import forms
from mayan.apps.templating.fields import ModelTemplateField

from ..models.workflow_instance_models import WorkflowInstance
from ..models.workflow_transition_models import WorkflowTransition


class WorkflowTransitionForm(forms.ModelForm):
    def __init__(self, workflow, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fields[
            'origin_state'
        ].queryset = self.fields[
            'origin_state'
        ].queryset.filter(workflow=workflow)

        self.fields[
            'destination_state'
        ].queryset = self.fields[
            'destination_state'
        ].queryset.filter(workflow=workflow)

        self.fields['condition'] = ModelTemplateField(
            initial_help_text=self.fields['condition'].help_text,
            label=self.fields['condition'].label, model=WorkflowInstance,
            model_variable='workflow_instance', required=False
        )

    class Meta:
        fields = ('label', 'origin_state', 'destination_state', 'condition')
        model = WorkflowTransition
