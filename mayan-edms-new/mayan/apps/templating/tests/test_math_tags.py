import math

from mayan.apps.testing.tests.base import BaseTestCase

from .mixins import TemplateTestMixin


class TemplateTagMathTestCase(TemplateTestMixin, BaseTestCase):
    def test_add_float(self):
        value_1 = 5.2
        value_2 = 2
        result = self._render_test_template(
            template_string='{{{{ {} | math_add:{} }}}}'.format(
                value_1, value_2
            )
        )
        self.assertEqual(
            result, str(value_1 + value_2)
        )

    def test_add_integer(self):
        value_1 = 5
        value_2 = 2
        result = self._render_test_template(
            template_string='{{{{ {} | math_add:{} }}}}'.format(
                value_1, value_2
            )
        )
        self.assertEqual(
            result, str(value_1 + value_2)
        )

    def test_absolute_float(self):
        value = -2.5
        result = self._render_test_template(
            template_string='{{{{ {} | math_absolute }}}}'.format(value)
        )
        self.assertEqual(
            result, str(
                abs(value)
            )
        )

    def test_absolute_integer(self):
        value = -1
        result = self._render_test_template(
            template_string='{{{{ {} | math_absolute }}}}'.format(value)
        )
        self.assertEqual(
            result, str(
                abs(value)
            )
        )

    def test_divide_float(self):
        value_1 = 2.2
        value_2 = 2
        result = self._render_test_template(
            template_string='{{{{ {} | math_divide:{} }}}}'.format(
                value_1, value_2
            )
        )
        self.assertEqual(
            result, str(
                value_1 / value_2
            )
        )

    def test_divide_integer(self):
        value_1 = 5
        value_2 = 2
        result = self._render_test_template(
            template_string='{{{{ {} | math_divide:{} }}}}'.format(
                value_1, value_2
            )
        )
        self.assertEqual(
            result, str(
                value_1 / value_2
            )
        )

    def test_exponentiate_float(self):
        value_1 = 4.2
        value_2 = 2
        result = self._render_test_template(
            template_string='{{{{ {} | math_exponentiate:{} }}}}'.format(
                value_1, value_2
            )
        )
        self.assertEqual(
            result, str(
                value_1 ** value_2
            )
        )

    def test_exponentiate_integer(self):
        value_1 = 2
        value_2 = 2
        result = self._render_test_template(
            template_string='{{{{ {} | math_exponentiate:{} }}}}'.format(
                value_1, value_2
            )
        )
        self.assertEqual(
            result, str(
                value_1 ** value_2
            )
        )

    def test_math_floor_divide_float(self):
        value_1 = 20.2
        value_2 = 2
        result = self._render_test_template(
            template_string='{{{{ {} | math_floor_divide:{} }}}}'.format(
                value_1, value_2
            )
        )
        self.assertEqual(
            result, str(
                value_1 // value_2
            )
        )

    def test_math_floor_divide_integer(self):
        value_1 = 10
        value_2 = 2
        result = self._render_test_template(
            template_string='{{{{ {} | math_floor_divide:{} }}}}'.format(
                value_1, value_2
            )
        )
        self.assertEqual(
            result, str(
                value_1 // value_2
            )
        )

    def test_math_modulo_float(self):
        value_1 = 20.2
        value_2 = 2
        result = self._render_test_template(
            template_string='{{{{ {} | math_modulo:{} }}}}'.format(
                value_1, value_2
            )
        )
        self.assertEqual(
            result, str(
                value_1 % value_2
            )
        )

    def test_math_modulo_integer(self):
        value_1 = 10
        value_2 = 2
        result = self._render_test_template(
            template_string='{{{{ {} | math_modulo:{} }}}}'.format(
                value_1, value_2
            )
        )
        self.assertEqual(
            result, str(
                value_1 % value_2
            )
        )

    def test_math_multiply_float(self):
        value_1 = 10.5
        value_2 = 2
        result = self._render_test_template(
            template_string='{{{{ {} | math_multiply:{} }}}}'.format(
                value_1, value_2
            )
        )
        self.assertEqual(
            result, str(
                value_1 * value_2
            )
        )

    def test_math_multiply_integer(self):
        value_1 = 20
        value_2 = 2
        result = self._render_test_template(
            template_string='{{{{ {} | math_multiply:{} }}}}'.format(
                value_1, value_2
            )
        )
        self.assertEqual(
            result, str(
                value_1 * value_2
            )
        )

    def test_square_root_float(self):
        value = 10.5
        result = self._render_test_template(
            template_string='{{{{ {} | math_square_root }}}}'.format(value)
        )
        self.assertEqual(
            result, str(
                math.sqrt(value)
            )
        )

    def test_square_root_integer(self):
        value = 16
        result = self._render_test_template(
            template_string='{{{{ {} | math_square_root }}}}'.format(value)
        )
        self.assertEqual(
            result, str(
                math.sqrt(value)
            )
        )

    def test_math_substract_float(self):
        value_1 = 50
        value_2 = 2
        result = self._render_test_template(
            template_string='{{{{ {} | math_substract:{} }}}}'.format(
                value_1, value_2
            )
        )
        self.assertEqual(
            result, str(
                value_1 - value_2
            )
        )

    def test_math_substract_integer(self):
        value_1 = 60.5
        value_2 = 2
        result = self._render_test_template(
            template_string='{{{{ {} | math_substract:{} }}}}'.format(
                value_1, value_2
            )
        )
        self.assertEqual(
            result, str(
                value_1 - value_2
            )
        )
