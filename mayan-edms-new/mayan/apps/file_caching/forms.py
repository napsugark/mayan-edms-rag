from django.utils.translation import gettext_lazy as _

from mayan.apps.forms import forms

from .models import Cache, CachePartition


class CacheDetailForm(forms.DetailForm):
    fieldsets = (
        (
            _(message='Identification'), {
                'fields': ('label',)
            }
        ), (
            _(message='Storage'), {
                'fields': ('defined_storage_name',)
            }
        ), (
            _(message='Configuration'), {
                'fields': (
                    'get_maximum_size_display', 'get_total_size_display'
                )
            }
        ), (
            _(message='Objects'), {
                'fields': ('get_partition_count', 'get_partition_file_count')
            }
        )
    )

    class Meta:
        extra_fields = (
            {'field': 'label'},
            {'field': 'get_maximum_size_display'},
            {'field': 'get_total_size_display'},
            {'field': 'get_partition_count'},
            {'field': 'get_partition_file_count'}
        )
        fields = ('defined_storage_name',)
        model = Cache


class CachePartitionDetailForm(forms.DetailForm):
    class Meta:
        fields = ('cache', 'name')
        model = CachePartition
