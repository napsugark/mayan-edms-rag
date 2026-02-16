import hashlib

from django.apps import apps
from django.core import serializers
from django.db import transaction
from django.utils.translation import gettext_lazy as _

from ..literals import GRAPHVIZ_SYMBOL_CONDITIONAL, GRAPHVIZ_SYMBOL_TRIGGER


class WorkflowTransitionBusinessLogicMixin:
    def do_diagram_generate(self, diagram):
        if self.has_condition():
            transition_label = '{} {}'.format(
                self.label, GRAPHVIZ_SYMBOL_CONDITIONAL
            )
        else:
            transition_label = self.label

        if self.trigger_events.exists():
            transition_label = '{} {}'.format(
                transition_label, GRAPHVIZ_SYMBOL_TRIGGER
            )

        edge_kwargs = {
            'head_name': self.destination_state.get_graph_id(),
            'label': '''<
                <table border="3" cellborder="0" color="white">
                    <tr>
                        <td bgcolor="white">{}</td>
                    </tr>
                </table>
            >'''.format(transition_label),
            'tail_name': self.origin_state.get_graph_id()
        }
        diagram.edge(**edge_kwargs)

    def do_execute(
        self, workflow_instance, comment=None, extra_data=None, user=None
    ):
        WorkflowInstanceLogEntry = apps.get_model(
            app_label='document_states', model_name='WorkflowInstanceLogEntry'
        )

        with transaction.atomic():
            workflow_instance.state_active = self.destination_state
            workflow_instance.save()

            workflow_instance_log_entry = WorkflowInstanceLogEntry(
                comment=comment, extra_data=extra_data, transition=self,
                user=user, workflow_instance=workflow_instance
            )
            workflow_instance_log_entry._event_actor = user
            workflow_instance_log_entry.save()

        return workflow_instance_log_entry

    def get_field_display(self):
        field_list = [
            str(field) for field in self.fields.all()
        ]
        field_list.sort()

        return ', '.join(field_list)

    get_field_display.short_description = _(message='Fields')

    def get_form_schema(self, schema, workflow_instance):
        for field in self.fields.all():
            field.get_form_schema(
                schema=schema, workflow_instance=workflow_instance
            )

    def get_hash(self):
        result = hashlib.sha256(
            string=serializers.serialize(
                format='json', queryset=(self,)
            ).encode()
        )
        for trigger_event in self.trigger_events.all():
            result.update(
                trigger_event.get_hash().encode()
            )

        for field in self.fields.all():
            result.update(
                field.get_hash().encode()
            )

        return result.hexdigest()
