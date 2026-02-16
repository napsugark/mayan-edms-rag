from datetime import datetime
import html

from django.template import TemplateSyntaxError

from mayan.apps.testing.tests.base import BaseTestCase

from .mixins import TemplateTestMixin


class TemplateFilterDateParseISOTestCase(TemplateTestMixin, BaseTestCase):
    def test_correct_format(self):
        result = self._render_test_template(
            template_string='{{ "1990-01-01T00:00"|date_parse_iso }}'
        )
        self.assertEqual(
            result, 'Jan. 1, 1990, midnight'
        )

    def test_incorrect_format(self):
        with self.assertRaises(expected_exception=TemplateSyntaxError):
            self._render_test_template(
                template_string='{{ "90-01-01T00:00"|date_parse_iso }}'
            )


class TemplateTagDateParseTestCase(TemplateTestMixin, BaseTestCase):
    def test_basic_functionality(self):
        result = self._render_test_template(
            template_string='{% date_parse "01/02/03 00:00:00+1" %}'
        )
        self.assertEqual(
            result, '2003-01-02 00:00:00+01:00'
        )

    def test_dayfirst_false(self):
        result = self._render_test_template(
            template_string='{% date_parse "01/02/03 00:00:00+1" dayfirst=False %}'
        )
        self.assertEqual(
            result, '2003-01-02 00:00:00+01:00'
        )

    def test_dayfirst_true(self):
        result = self._render_test_template(
            template_string='{% date_parse "01/02/03 00:00:00+1" dayfirst=True %}'
        )
        self.assertEqual(
            result, '2003-02-01 00:00:00+01:00'
        )

    def test_fuzzy_false(self):
        with self.assertRaises(expected_exception=TemplateSyntaxError):
            self._render_test_template(
                template_string='{% date_parse "Today is 01/02/03 00:00:00+1" fuzzy=False %}'
            )

    def test_fuzzy_true(self):
        result = self._render_test_template(
            template_string='{% date_parse "Today is 01/02/03 00:00:00+1" fuzzy=True %}'
        )
        self.assertEqual(
            result, '2003-01-02 00:00:00+01:00'
        )

    def test_fuzzy_with_tokens_false(self):
        with self.assertRaises(expected_exception=TemplateSyntaxError):
            self._render_test_template(
                template_string='{% date_parse "Today is 01/02/03 00:00:00+1" fuzzy_with_tokens=False as output %}{{ output.0 }}{{ output.1 }}'
            )

    def test_fuzzy_with_tokens_true(self):
        result = self._render_test_template(
            template_string='{% date_parse "Today is 01/02/03 00:00:00+1" fuzzy_with_tokens=True as output %}{{ output.0 }}'
        )
        self.assertEqual(
            result, 'Jan. 1, 2003, 11 p.m.'
        )
        result = self._render_test_template(
            template_string='{% date_parse "Today is 01/02/03 00:00:00+1" fuzzy_with_tokens=True as output %}{{ output.1 }}'
        )
        self.assertEqual(
            result, html.escape("('Today is ', ' ')")
        )

    def test_ignoretz_false(self):
        result = self._render_test_template(
            template_string='{% date_parse "01/02/03 00:00:00+1" ignoretz=False %}'
        )
        self.assertEqual(
            result, '2003-01-02 00:00:00+01:00'
        )

    def test_ignoretz_true(self):
        result = self._render_test_template(
            template_string='{% date_parse "01/02/03 00:00:00+1" ignoretz=True %}'
        )
        self.assertEqual(
            result, '2003-01-02 00:00:00'
        )

    def test_yearfirst_false(self):
        result = self._render_test_template(
            template_string='{% date_parse "01/02/03 00:00:00+1" yearfirst=False %}'
        )
        self.assertEqual(
            result, '2003-01-02 00:00:00+01:00'
        )

    def test_yearfirst_true(self):
        result = self._render_test_template(
            template_string='{% date_parse "01/02/03 00:00:00+1" yearfirst=True %}'
        )
        self.assertEqual(
            result, '2001-02-03 00:00:00+01:00'
        )


class TemplateTagTimeDeltaTestCase(TemplateTestMixin, BaseTestCase):
    def test_basic_functionality(self):
        now = datetime.now()

        result = self._render_test_template(
            template_string='{{% date_parse "{}" as date_object %}}{{% timedelta date_object days=366 as date_new %}}{{{{ date_new.year }}}}'.format(
                now.isoformat()
            )
        )
        self.assertEqual(
            result, str(now.year + 1)
        )
