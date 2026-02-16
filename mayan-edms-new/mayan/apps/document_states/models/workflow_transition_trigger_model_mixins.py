import hashlib

from django.core import serializers


class WorkflowTransitionTriggerEventBusinessLogicMixin:
    def get_hash(self):
        return hashlib.sha256(
            string=serializers.serialize(
                format='json', queryset=(self,)
            ).encode()
        ).hexdigest()
