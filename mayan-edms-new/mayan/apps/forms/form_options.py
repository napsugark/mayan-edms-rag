class FormOptions:
    def __init__(self, form, kwargs, options=None):
        """
        Option definitions will be iterated. The option value will be
        determined in the following order: as passed via keyword
        arguments during form initialization, as form get_... method or
        finally as static Meta options. This is to allow a form with
        Meta options or method to be overridden at initialization
        and increase the usability of a single class.
        """
        for name, default_value in self.option_definitions.items():
            try:
                # Check for a runtime value via kwargs
                value = kwargs.pop(name)
            except KeyError:
                try:
                    # Check if there is a get_... method
                    value = getattr(
                        self, 'get_{}'.format(name)
                    )()
                except AttributeError:
                    try:
                        # Check the meta class options
                        value = getattr(options, name)
                    except AttributeError:
                        value = default_value

            setattr(self, name, value)


class DetailFormOption(FormOptions):
    # Dictionary list of option names and default values.
    option_definitions = {
        'extra_fields': []
    }


class FilteredSelectionFormOptions(FormOptions):
    # Dictionary list of option names and default values.
    option_definitions = {
        'allow_multiple': False,
        'field_name': None,
        'help_text': None,
        'label': None,
        'model': None,
        'permission': None,
        'queryset': None,
        'required': True,
        'user': None,
        'widget_attributes': {'size': '10'},
        'widget_class': None
    }
