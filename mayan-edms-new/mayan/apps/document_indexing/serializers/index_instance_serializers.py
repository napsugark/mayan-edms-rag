from django.utils.translation import gettext_lazy as _

from rest_framework.reverse import reverse

from mayan.apps.rest_api import serializers

from ..models import IndexInstance, IndexInstanceNode


class IndexInstanceSerializer(serializers.ModelSerializer):
    depth = serializers.IntegerField(
        label=_(message='Depth'), read_only=True, source='get_level_count'
    )
    node_count = serializers.IntegerField(
        label=_(message='Node count'), read_only=True,
        source='get_descendants_count'
    )
    nodes_url = serializers.SerializerMethodField(
        label=_(message='Nodes URL'), read_only=True
    )
    url = serializers.SerializerMethodField(
        label=_(message='URL'), read_only=True
    )

    class Meta:
        fields = ('depth', 'label', 'id', 'nodes_url', 'node_count', 'url')
        model = IndexInstance
        read_only_fields = fields

    def get_url(self, obj):
        return reverse(
            format=self.context['format'], kwargs={
                'index_instance_id': obj.pk
            }, request=self.context['request'],
            viewname='rest_api:indexinstance-detail'
        )

    def get_nodes_url(self, obj):
        return reverse(
            format=self.context['format'], kwargs={
                'index_instance_id': obj.pk
            }, request=self.context['request'],
            viewname='rest_api:indexinstancenode-list'
        )


class IndexInstanceNodeSerializer(serializers.ModelSerializer):
    children_url = serializers.SerializerMethodField(
        label=_(message='Children URL'), read_only=True
    )
    depth = serializers.IntegerField(
        label=_(message='Depth'), read_only=True, source='get_level_count'
    )
    documents_url = serializers.SerializerMethodField(
        label=_(message='Documents URL'), read_only=True
    )
    index_url = serializers.SerializerMethodField(
        label=_(message='Index URL'), read_only=True
    )
    node_count = serializers.IntegerField(
        label=_(message='Node count'), read_only=True,
        source='get_descendants_count'
    )
    parent_url = serializers.SerializerMethodField(
        label=_(message='Parent URL'), read_only=True
    )
    url = serializers.SerializerMethodField(
        label=_(message='URL'), read_only=True
    )

    class Meta:
        fields = (
            'depth', 'documents_url', 'children_url', 'id', 'index_url',
            'level', 'node_count', 'parent_id', 'parent_url', 'value', 'url'
        )
        model = IndexInstanceNode
        read_only_fields = fields

    def get_children_url(self, obj):
        return reverse(
            format=self.context['format'], kwargs={
                'index_instance_id': obj.index().pk,
                'index_instance_node_id': obj.pk
            }, request=self.context['request'],
            viewname='rest_api:indexinstancenode-children-list'
        )

    def get_documents_url(self, obj):
        return reverse(
            format=self.context['format'], kwargs={
                'index_instance_id': obj.index().pk,
                'index_instance_node_id': obj.pk
            }, request=self.context['request'],
            viewname='rest_api:indexinstancenode-document-list'
        )

    def get_index_url(self, obj):
        return reverse(
            format=self.context['format'], kwargs={
                'index_instance_id': obj.index().pk
            }, request=self.context['request'],
            viewname='rest_api:indexinstance-detail'
        )

    def get_parent_url(self, obj):
        if obj.parent and not obj.parent.is_root_node():
            return reverse(
                format=self.context['format'], kwargs={
                    'index_instance_id': obj.index().pk,
                    'index_instance_node_id': obj.parent.pk
                }, request=self.context['request'],
                viewname='rest_api:indexinstancenode-detail'
            )
        else:
            return ''

    def get_url(self, obj):
        return reverse(
            format=self.context['format'], kwargs={
                'index_instance_id': obj.index().pk,
                'index_instance_node_id': obj.pk
            }, request=self.context['request'],
            viewname='rest_api:indexinstancenode-detail'
        )
