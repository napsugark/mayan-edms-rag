from django.utils.html import conditional_escape
from django.utils.safestring import mark_safe
from django.utils.translation import gettext_lazy as _, gettext

from mayan.apps.forms import form_fields, form_widgets, forms

from .models import DocumentVersionPageOCRContent


class DocumentVersionPageOCRContentDetailForm(forms.Form):
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
        page = kwargs.pop('instance', None)
        super().__init__(*args, **kwargs)
        content = ''
        self.fields['contents'].initial = ''

        try:
            page_content = page.ocr_content.content
        except DocumentVersionPageOCRContent.DoesNotExist:
            """
            Not critical, just ignore and display empty content.
            """
        else:
            content = conditional_escape(
                str(page_content)
            )

        self.fields['contents'].initial = mark_safe(s=content)


class DocumentVersionPageOCRContentEditForm(forms.ModelForm):
    content = form_fields.CharField(
        label=_(message='Contents'),
        widget=form_widgets.Textarea(
            attrs={
                'class': 'full-height',
                'data-height-difference': 360
            }
        )
    )

    class Meta:
        fields = ('content',)
        model = DocumentVersionPageOCRContent


class DocumentVersionOCRContentForm(forms.Form):
    """
    Form that concatenates all of a document pages' text content into a
    single textarea widget
    """
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
                page_content = page.ocr_content.content
            except DocumentVersionPageOCRContent.DoesNotExist:
                """
                Not critical, just ignore and proceed to next page.
                """
            else:
                content.append(
                    conditional_escape(
                        text=str(page_content)
                    )
                )
                content.append(
                    '\n\n\n<hr/><div class="document-page-content-divider">- %s -</div><hr/>\n\n\n' % (
                        gettext(
                            'Page %(page_number)d'
                        ) % {'page_number': page.page_number}
                    )
                )

        self.fields['contents'].initial = mark_safe(
            s=''.join(content)
        )
