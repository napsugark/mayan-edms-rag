import hashlib

from django.core import serializers
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

from mayan.apps.common.serialization import yaml_load
from mayan.apps.common.utils import comma_splitter
from mayan.apps.forms.literals import EMPTY_LABEL
from mayan.apps.templating.template_backends import Template

from ..literals import FIELD_TYPE_MAPPING, WIDGET_CLASS_MAPPING


class WorkflowTransitionFieldBusinessLogicMixin:
    def get_default_value(self, workflow_instance):
        template = Template(template_string=self.default)
        context = {'workflow_instance': workflow_instance}

        return template.render(context=context)

    def get_form_schema(self, schema, workflow_instance):
        schema['fields'][self.name] = {
            'class': FIELD_TYPE_MAPPING[self.field_type],
            'help_text': self.help_text,
            'label': self.label,
            'required': self.required
        }

        schema_field = schema['fields'][self.name]

        if self.widget:
            schema['widgets'][self.name] = {
                'class': WIDGET_CLASS_MAPPING[self.widget],
                'kwargs': self.get_widget_kwargs()
            }

        if self.default:
            default_value = self.get_default_value(
                workflow_instance=workflow_instance
            )
            schema_field['default'] = default_value

        if self.lookup:
            try:
                lookup_choices = self.get_lookup_values(
                    workflow_instance=workflow_instance
                )

                field_choices = list(
                    zip(lookup_choices, lookup_choices)
                )

                if not self.required:
                    field_choices.insert(
                        0, ('', EMPTY_LABEL)
                    )

                schema_field['class'] = 'django.forms.fields.ChoiceField'
                schema_field['kwargs'] = {'choices': field_choices}

                schema['widgets'][self.name] = {
                    'class': 'django.forms.fields.Select',
                    'kwargs': {
                        'attrs': {'class': 'select2'}
                    }
                }
            except Exception as exception:
                schema_field['class'] = 'django.forms.fields.CharField'
                schema_field['kwargs'] = {
                    'initial': _(
                        message='Lookup value error: %s'
                    ) % exception
                }
                schema['widgets'][self.name] = {
                    'class': 'django.forms.fields.TextInput',
                    'kwargs': {
                        'attrs': {'readonly': 'readonly'}
                    }
                }

    def get_hash(self):
        return hashlib.sha256(
            string=serializers.serialize(
                format='json', queryset=(self,)
            ).encode()
        ).hexdigest()

    def get_lookup_values(self, workflow_instance):
        template = Template(
            context_entry_name_list=('groups', 'users'),
            template_string=self.lookup
        )
        return comma_splitter(
            template.render(
                context={'workflow_instance': workflow_instance}
            )
        )

    def get_widget_kwargs(self):
        return yaml_load(
            stream=self.widget_kwargs or '{}'
        )

    def has_default(self):
        if self.default:
            return True
        else:
            return False

    has_default.short_description = _(message='Has a default?')

    def has_lookup(self):
        if self.lookup:
            return True
        else:
            return False

    has_lookup.short_description = _(message='Has a lookup?')

    def validate_value(self, value, workflow_instance):
        if self.lookup:
            lookup_options = self.get_lookup_values(
                workflow_instance=workflow_instance
            )

            if value and value not in lookup_options:
                raise ValidationError(
                    message=_(
                        message='Value is not one of the provided options.'
                    )
                )
