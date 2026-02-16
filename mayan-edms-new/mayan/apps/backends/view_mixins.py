class ViewMixinDynamicFormBackendClass:
    def get_form_schema(self):
        backend_class = self.get_backend_class()

        form_schema = backend_class.get_form_schema(
            **self.get_form_schema_extra_kwargs()
        )

        return form_schema

    def get_form_schema_extra_kwargs(self):
        return {}
