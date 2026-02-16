class SourceBackendActionInterfaceArgument:
    def __eq__(self, other):
        # Allow quick deduplication by using `not in list`.
        return self.name == other.name

    def __init__(self, help_text=None, hidden=False, required=True, **kwargs):
        try:
            self.default = kwargs.pop('default')
        except KeyError:
            self.has_default = False
        else:
            self.has_default = True

        self.help_text = help_text
        self.hidden = hidden
        self.required = required
