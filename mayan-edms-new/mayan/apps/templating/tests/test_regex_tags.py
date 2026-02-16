from mayan.apps.testing.tests.base import BaseTestCase

from .mixins import TemplateTestMixin


class TemplateTagRegexTestCase(TemplateTestMixin, BaseTestCase):
    def test_tag_regex_findall_false(self):
        result = self._render_test_template(
            template_string='{% regex_findall "\\d" "abcxyz" as result %}{% if result %}{{ result }}{% endif %}'
        )
        self.assertEqual(result, '')

    def test_tag_regex_findall_true(self):
        result = self._render_test_template(
            template_string='{% regex_findall "\\d" "abc123" as result %}{{ result.0 }}{{ result.1 }}{{ result.2 }}'
        )
        self.assertEqual(result, '123')

    def test_tag_regex_match_false(self):
        result = self._render_test_template(
            template_string='{% regex_match "\\d" "abc123" as result %}{% if result %}{{ result }}{% endif %}'
        )
        self.assertEqual(result, '')

    def test_tag_regex_match_true(self):
        result = self._render_test_template(
            template_string='{% regex_match "\\d" "123abc" as result %}{% if result %}{{ result.0 }}{% endif %}'
        )
        self.assertEqual(result, '1')

    def test_tag_regex_search_false(self):
        result = self._render_test_template(
            template_string='{% regex_search "\\d" "abcxyz" as result %}{% if result %}{{ result }}{% endif %}'
        )
        self.assertEqual(result, '')

    def test_tag_regex_search_true(self):
        result = self._render_test_template(
            template_string='{% regex_search "\\d" "abc123" as result %}{{ result.0 }}'
        )
        self.assertEqual(result, '1')

    def test_tag_regex_sub_false(self):
        result = self._render_test_template(
            template_string='{% regex_sub "\\d" "XX" "abcxyz" as result %}{{ result }}'
        )
        self.assertEqual(result, 'abcxyz')

    def test_tag_regex_sub_true(self):
        result = self._render_test_template(
            template_string='{% regex_sub "\\d" "XX" "abc123" as result %}{{ result }}'
        )
        self.assertEqual(result, 'abcXXXXXX')
