import html
import json

from mayan.apps.testing.tests.base import BaseTestCase

from .mixins import TemplateTestMixin


class TemplateTagJSONTestCase(TemplateTestMixin, BaseTestCase):
    def test_json(self):
        value = {'key': 'value'}
        value_json = json.dumps(obj=value)

        template_string = '{{{{ \'{}\' | json_load }}}}'.format(value_json)
        result = self._render_test_template(template_string=template_string)
        self.assertEqual(
            result, html.escape(
                str(value)
            )
        )
