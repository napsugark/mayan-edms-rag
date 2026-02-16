from django.utils.translation import gettext_lazy as _

from mayan.apps.rest_api import serializers
from mayan.apps.rest_api.relations import FilteredPrimaryKeyRelatedField

from .models import Tag
from .permissions import permission_tag_attach, permission_tag_remove


class TagSerializer(serializers.HyperlinkedModelSerializer):
    documents_url = serializers.HyperlinkedIdentityField(
        label=_(message='Documents URL'), lookup_url_kwarg='tag_id',
        view_name='rest_api:tag-document-list'
    )

    class Meta:
        extra_kwargs = {
            'url': {
                'label': _(message='URL'),
                'lookup_url_kwarg': 'tag_id',
                'view_name': 'rest_api:tag-detail'
            }
        }
        fields = (
            'color', 'documents_url', 'id', 'label', 'url'
        )
        model = Tag


class DocumentTagAttachSerializer(serializers.Serializer):
    tag = FilteredPrimaryKeyRelatedField(
        help_text=_(
            message='Primary key of the tag to add to the document.'
        ), label=_(message='Tag ID'), source_model=Tag,
        source_permission=permission_tag_attach
    )


class DocumentTagRemoveSerializer(serializers.Serializer):
    tag = FilteredPrimaryKeyRelatedField(
        help_text=_(
            message='Primary key of the tag to remove from the document.'
        ), label=_(message='Tag ID'), source_model=Tag,
        source_permission=permission_tag_remove
    )
