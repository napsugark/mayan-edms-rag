from django.utils.encoding import force_str
from django.utils.html import conditional_escape
from django.utils.safestring import mark_safe
from django.utils.translation import gettext_lazy as _, gettext

from mayan.apps.forms import form_fields, form_widgets, forms

from .models import DocumentFilePageContent


class DocumentFileContentForm(forms.Form):
    """
    Form that concatenates all of a document pages' text content into a
    single textarea widget
    """

    def __init__(self, *args, **kwargs):
        self.document = kwargs.pop('instance', None)
        super().__init__(*args, **kwargs)
        content = []
        self.fields['contents'].initial = ''
        try:
            document_pages = self.document.pages.all()
        except AttributeError:
            document_pages = []

        for page in document_pages:
            try:
                page_content = page.content.content
            except DocumentFilePageContent.DoesNotExist:
                """Page does not have parsed content, skip."""
            else:
                content.append(
                    conditional_escape(
                        text=force_str(s=page_content)
                    )
                )
                content.append(
                    '\n\n\n<hr/><div class="document-page-content-divider">- %s -</div><hr/>\n\n\n' % (
                        gettext(
                            message='Page %(page_number)d'
                        ) % {'page_number': page.page_number}
                    )
                )

        self.fields['contents'].initial = mark_safe(
            s=''.join(content)
        )

    contents = form_fields.CharField(
        label=_(message='Contents'),
        widget=form_widgets.TextAreaDiv(
            attrs={
                'class': 'full-height',
                'data-height-difference': 360
            }
        )
    )


class DocumentFilePageContentForm(forms.Form):
    contents = form_fields.CharField(
        label=_(message='Contents'),
        widget=form_widgets.TextAreaDiv(
            attrs={
                'class': 'full-height',
                'data-height-difference': 360
            }
        )
    )

    def __init__(self, *args, **kwargs):
        document_page = kwargs.pop('instance', None)
        super().__init__(*args, **kwargs)
        content = ''
        self.fields['contents'].initial = ''

        try:
            page_content = document_page.content.content
        except DocumentFilePageContent.DoesNotExist:
            pass
        else:
            content = conditional_escape(
                text=force_str(s=page_content)
            )

        self.fields['contents'].initial = mark_safe(s=content)
