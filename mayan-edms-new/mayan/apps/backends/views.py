from django.http import Http404

from mayan.apps.views.generics import (
    SingleObjectDynamicFormCreateView, SingleObjectDynamicFormEditView
)

from .view_mixins import ViewMixinDynamicFormBackendClass


class ViewSingleObjectDynamicFormModelBackendCreate(
    ViewMixinDynamicFormBackendClass, SingleObjectDynamicFormCreateView
):
    def get_backend_class(self):
        try:
            return self.backend_class.get(
                name=self.kwargs['backend_path']
            )
        except KeyError:
            raise Http404(
                '{} class not found'.format(
                    self.kwargs['backend_path']
                )
            )


class ViewSingleObjectDynamicFormModelBackendEdit(
    ViewMixinDynamicFormBackendClass, SingleObjectDynamicFormEditView
):
    def get_backend_class(self):
        return self.object.get_backend_class()
