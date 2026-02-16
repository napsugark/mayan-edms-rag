from mayan.apps.navigation.column_widgets import SourceColumnWidget


class WorkflowLogExtraDataWidget(SourceColumnWidget):
    template_name = 'document_states/extra_data.html'


class WorkflowTemplateTransitionTriggerColumnWidget(SourceColumnWidget):
    template_name = 'document_states/column_widgets/transition_triggers.html'
