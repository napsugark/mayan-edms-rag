from django.http import Http404
from django.utils.translation import gettext_lazy as _

from .classes import StatisticType


class StatisticTypeViewMixin:
    def dispatch(self, request, *args, **kwargs):
        self.object = self.get_object()
        return super().dispatch(request=request, *args, **kwargs)

    def get_object(self):
        try:
            return StatisticType.get(
                slug=self.kwargs['slug']
            )
        except KeyError:
            raise Http404(
                _(message='Statistic "%s" not found.') % self.kwargs['slug']
            )
