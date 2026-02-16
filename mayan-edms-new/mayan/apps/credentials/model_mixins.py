class StoredCredentialBusinessLogicMixin:
    def get_backend_data(self):
        obj = super().get_backend_data()

        if self.backend_path:
            backend_class = self.get_backend_class()

            if hasattr(backend_class, 'post_processing'):
                obj = backend_class.post_processing(obj=obj)

        return obj
