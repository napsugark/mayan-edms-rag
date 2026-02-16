import hashlib

from django.core import serializers
from django.utils.translation import gettext_lazy as _

from ..literals import GRAPHVIZ_SYMBOL_CONDITIONAL, GRAPHVIZ_SYMBOL_TIME


class WorkflowStateEscalationBusinessLogicMixin:
    def do_diagram_generate(self, diagram):
        if self.enabled:
            escalation_label = '{} {}'.format(
                GRAPHVIZ_SYMBOL_TIME, self.get_time_display()
            )
            if self.has_condition():
                escalation_label = '{} {}'.format(
                    escalation_label, GRAPHVIZ_SYMBOL_CONDITIONAL
                )

            edge_kwargs = {
                'head_name': self.transition.destination_state.get_graph_id(),
                'label': '''<
                    <table border="3" cellborder="0" color="white">
                        <tr>
                            <td bgcolor="white">{}</td>
                        </tr>
                    </table>
                >'''.format(escalation_label,),
                'style': 'dashed',
                'tail_name': self.state.get_graph_id()
            }
            diagram.edge(**edge_kwargs)

    def get_comment(self):
        return self.comment or _(message='Workflow escalation.')

    def get_hash(self):
        return hashlib.sha256(
            string=serializers.serialize(
                format='json', queryset=(self,)
            ).encode()
        ).hexdigest()

    def get_time_display(self):
        return '{} {}'.format(self.amount, self.unit)
