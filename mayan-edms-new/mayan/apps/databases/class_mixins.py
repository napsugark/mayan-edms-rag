from django.utils.translation import gettext_lazy as _

from mayan.apps.templating.template_backends import Template


class MixinConditionTemplate:
    def evaluate_condition(self, context=None):
        if self.has_condition():
            condition_context = context or self.get_condition_context()
            condition_template = self.get_condition_template()

            template = Template(template_string=condition_template)
            result = template.render(context=condition_context)
            result = result.strip()

            return result
        else:
            return True

    def get_condition_context(self):
        return {}

    def get_condition_template(self):
        return self.condition

    def has_condition(self):
        condition_template = self.get_condition_template()

        if condition_template:
            return True
        else:
            return False
    has_condition.help_text = _(
        message='The object will be executed, depending on the condition '
        'return value.'
    )
    has_condition.short_description = _(message='Has a condition?')
