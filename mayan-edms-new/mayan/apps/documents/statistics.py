from django.apps import apps
from django.db.models import Count
from django.utils import timezone
from django.utils.translation import gettext_lazy as _

import qsstats

from mayan.apps.mayan_statistics.classes import (
    StatisticNamespace, StatisticTypeDoughnutChart, StatisticTypeLineChart
)

from .literals import MONTH_NAMES
from .permissions import permission_document_view


def get_month_name(month_number):
    return str(
        MONTH_NAMES[month_number - 1]
    )


def new_documents_per_month():
    Document = apps.get_model(app_label='documents', model_name='Document')

    qss = qsstats.QuerySetStats(
        Document.valid.all(), 'datetime_created'
    )

    now = timezone.now().date()
    start = timezone.datetime(year=now.year, month=1, day=1).date()

    return {
        'series': {
            'Documents': map(
                lambda x: {
                    get_month_name(month_number=x[0].month): x[1]
                }, qss.time_series(start=start, end=now, interval='months')
            )
        }
    }


def new_document_pages_per_month():
    DocumentFilePage = apps.get_model(
        app_label='documents', model_name='DocumentFilePage'
    )

    qss = qsstats.QuerySetStats(
        DocumentFilePage.valid.all(), 'document_file__document__datetime_created'
    )

    now = timezone.now().date()
    start = timezone.datetime(year=now.year, month=1, day=1).date()

    return {
        'series': {
            'Pages': map(
                lambda x: {
                    get_month_name(month_number=x[0].month): x[1]
                }, qss.time_series(start=start, end=now, interval='months')
            )
        }
    }


def new_documents_this_month(user=None):
    AccessControlList = apps.get_model(
        app_label='acls', model_name='AccessControlList'
    )
    Document = apps.get_model(app_label='documents', model_name='Document')

    queryset = Document.valid.all()

    if user:
        queryset = AccessControlList.objects.restrict_queryset(
            permission=permission_document_view, user=user,
            queryset=queryset
        )

    qss = qsstats.QuerySetStats(queryset, 'datetime_created')
    return qss.this_month() or '0'


def new_document_files_per_month():
    DocumentFile = apps.get_model(
        app_label='documents', model_name='DocumentFile'
    )

    qss = qsstats.QuerySetStats(
        DocumentFile.valid.all(), 'document__datetime_created'
    )

    now = timezone.now().date()
    start = timezone.datetime(year=now.year, month=1, day=1).date()

    return {
        'series': {
            'Files': map(
                lambda x: {
                    get_month_name(month_number=x[0].month): x[1]
                }, qss.time_series(start=start, end=now, interval='months')
            )
        }
    }


def new_document_pages_this_month(user=None):
    AccessControlList = apps.get_model(
        app_label='acls', model_name='AccessControlList'
    )
    DocumentFilePage = apps.get_model(
        app_label='documents', model_name='DocumentFilePage'
    )

    queryset = DocumentFilePage.valid.all()

    if user:
        queryset = AccessControlList.objects.restrict_queryset(
            permission=permission_document_view, user=user,
            queryset=queryset
        )

    qss = qsstats.QuerySetStats(
        queryset, 'document_file__document__datetime_created'
    )
    return qss.this_month() or '0'


def total_document_per_month():
    Document = apps.get_model(app_label='documents', model_name='Document')

    qss = qsstats.QuerySetStats(Document.valid.all(), 'datetime_created')
    now = timezone.now()

    result = []

    for month in range(1, now.month + 1):
        next_month = month + 1

        if month == 12:
            next_month = 1
            year = now.year + 1
        else:
            next_month = month + 1
            year = now.year

        result.append(
            {
                get_month_name(month_number=month): qss.until(
                    timezone.datetime(year, next_month, 1, tzinfo=now.tzinfo)
                )
            }
        )

    return {
        'series': {
            'Documents': result
        }
    }


def total_document_file_per_month():
    DocumentFile = apps.get_model(
        app_label='documents', model_name='DocumentFile'
    )

    qss = qsstats.QuerySetStats(
        DocumentFile.valid.all(), 'document__datetime_created'
    )
    now = timezone.now()

    result = []

    for month in range(1, now.month + 1):
        next_month = month + 1

        if month == 12:
            next_month = 1
            year = now.year + 1
        else:
            next_month = month + 1
            year = now.year

        result.append(
            {
                get_month_name(month_number=month): qss.until(
                    timezone.datetime(year, next_month, 1, tzinfo=now.tzinfo)
                )
            }
        )

    return {
        'series': {
            'Files': result
        }
    }


def total_document_page_per_month():
    DocumentFilePage = apps.get_model(
        app_label='documents', model_name='DocumentFilePage'
    )

    qss = qsstats.QuerySetStats(
        DocumentFilePage.valid.all(),
        'document_file__document__datetime_created'
    )
    now = timezone.now()

    result = []

    for month in range(1, now.month + 1):
        next_month = month + 1

        if month == 12:
            next_month = 1
            year = now.year + 1
        else:
            next_month = month + 1
            year = now.year

        result.append(
            {
                get_month_name(month_number=month): qss.until(
                    timezone.datetime(year, next_month, 1, tzinfo=now.tzinfo)
                )
            }
        )

    return {
        'series': {
            'Pages': result
        }
    }


def statistic_document_count_per_document_type():
    DocumentType = apps.get_model(
        app_label='documents', model_name='DocumentType'
    )

    return {
        'series': {
            'document_types': tuple(
                DocumentType.objects.annotate(
                    value=Count('documents')
                ).values('label', 'value')
            )
        }
    }


def statistic_document_file_count_per_document_type():
    DocumentType = apps.get_model(
        app_label='documents', model_name='DocumentType'
    )

    return {
        'series': {
            'document_types': tuple(
                DocumentType.objects.annotate(
                    value=Count('documents__files')
                ).values('label', 'value')
            )
        }
    }


def statistic_document_file_page_count_per_document_type():
    DocumentType = apps.get_model(
        app_label='documents', model_name='DocumentType'
    )

    return {
        'series': {
            'document_types': tuple(
                DocumentType.objects.annotate(
                    value=Count('documents__files__file_pages')
                ).values('label', 'value')
            )
        }
    }


namespace = StatisticNamespace(
    slug='documents', label=_(message='Documents')
)
namespace.add_statistic(
    klass=StatisticTypeLineChart,
    slug='new-documents-per-month',
    label=_(message='New documents per month'),
    func=new_documents_per_month,
    minute='0'
)
namespace.add_statistic(
    klass=StatisticTypeLineChart,
    slug='new-document-files-per-month',
    label=_(message='New document files per month'),
    func=new_document_files_per_month,
    minute='0'
)
namespace.add_statistic(
    klass=StatisticTypeLineChart,
    slug='new-document-pages-per-month',
    label=_(message='New document pages per month'),
    func=new_document_pages_per_month,
    minute='0'
)
namespace.add_statistic(
    klass=StatisticTypeLineChart,
    slug='total-documents-at-each-month',
    label=_(message='Total documents at each month'),
    func=total_document_per_month,
    minute='0'
)
namespace.add_statistic(
    klass=StatisticTypeLineChart,
    slug='total-document-files-at-each-month',
    label=_(message='Total document files at each month'),
    func=total_document_file_per_month,
    minute='0'
)
namespace.add_statistic(
    klass=StatisticTypeLineChart,
    slug='total-document-pages-at-each-month',
    label=_(message='Total document pages at each month'),
    func=total_document_page_per_month,
    minute='0'
)
namespace.add_statistic(
    klass=StatisticTypeDoughnutChart,
    slug='document-count-per-document-type',
    label=_(message='Total documents per document type'),
    func=statistic_document_count_per_document_type,
    minute='0'
)
namespace.add_statistic(
    klass=StatisticTypeDoughnutChart,
    slug='document-file-count-per-document-type',
    label=_(message='Total document files per document type'),
    func=statistic_document_file_count_per_document_type,
    minute='0'
)
namespace.add_statistic(
    klass=StatisticTypeDoughnutChart,
    slug='document-file-page-count-per-document-type',
    label=_(message='Total document file pages per document type'),
    func=statistic_document_file_page_count_per_document_type,
    minute='0'
)
