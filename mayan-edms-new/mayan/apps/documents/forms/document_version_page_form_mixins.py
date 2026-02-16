class FormSetExtraFormKwargsMixin:
    def __init__(self, *args, **kwargs):
        self.form_extra_kwargs = kwargs.pop(
            'form_extra_kwargs', {}
        )
        super().__init__(*args, **kwargs)

    def get_form_kwargs(self, index):
        form_kwargs = super().get_form_kwargs(index=index)
        form_kwargs.update(self.form_extra_kwargs)
        return form_kwargs
