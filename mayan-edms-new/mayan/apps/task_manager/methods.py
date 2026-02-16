from django.utils.module_loading import import_string


def factory_method_periodic_task_save(super_save):
    def method_save(self, *args, **kwargs):
        import_string(dotted_path=self.task)
        return super_save(self, *args, **kwargs)
    return method_save
