import base64
import io

from django.utils.encoding import force_str

from mayan.apps.forms import form_widgets


class Base64ImageWidget(form_widgets.Widget):
    template_name = 'converter/forms/widgets/base64_image.html'

    def format_value(self, value):
        if value == '' or value is None:
            return None
        else:
            with io.BytesIO() as output:
                value.save(format='PNG', stream=output)
                image = output.getvalue()
                url = 'data:image/png;charset=utf-8;base64,{}'.format(
                    force_str(
                        s=base64.b64encode(s=image)
                    )
                )

                return url


class ImageWidget(form_widgets.Widget):
    template_name = 'converter/forms/widgets/widget_image.html'

    def format_value(self, value):
        if value == '' or value is None:
            return None
        return value
