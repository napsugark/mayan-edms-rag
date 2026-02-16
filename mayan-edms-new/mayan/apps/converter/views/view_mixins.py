from mayan.apps.forms import form_fields, forms

from ..classes import Layer
from ..forms import LayerTransformationForm


class ViewMixinDynamicTransformationFormClass:
    def get_form_class(self):
        transformation_class = self.get_transformation_class()

        TransformationForm = transformation_class.get_form_class()

        transformation_has_form_or_template = TransformationForm or transformation_class.get_template_name()

        if not transformation_has_form_or_template:
            # Transformation does not specify a form and does not have
            # a template either. Create a dynamic form based on the argument
            # list.
            class TransformationForm(forms.Form):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    for argument in transformation_class.get_arguments():
                        self.fields[argument] = form_fields.CharField(
                            required=False
                        )

        class MergedTransformationForm(
            TransformationForm, LayerTransformationForm
        ):
            """Model form merged with the specific transformation fields."""
            view = self

        return MergedTransformationForm


class ViewMixinLayer:
    def dispatch(self, request, *args, **kwargs):
        self.layer = self.get_layer()
        return super().dispatch(request=request, *args, **kwargs)

    def get_layer(self):
        return Layer.get(
            name=self.kwargs['layer_name']
        )


class ViewMixinTransformationTemplateName:
    def get_template_names(self):
        transformation_template_name = self.get_transformation_template_name()

        if transformation_template_name is None:
            transformation_template_name = self.template_name

        return (transformation_template_name,)

    def get_transformation_template_name(self):
        transformation_class = self.get_transformation_class()

        return transformation_class.get_template_name()
