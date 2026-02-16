from django.apps import apps
from django.core.management.base import BaseCommand

from ...services import send_document_version_to_rag


class Command(BaseCommand):
    help = 'Sync unsynced document versions to the external RAG service.'

    def handle(self, *args, **options):
        DocumentVersion = apps.get_model(
            app_label='documents', model_name='DocumentVersion'
        )
        RAGDocumentVersionSync = apps.get_model(
            app_label='rag_integration',
            model_name='RAGDocumentVersionSync'
        )

        # Get IDs of already synced document versions.
        synced_version_ids = RAGDocumentVersionSync.objects.filter(
            rag_synced=True
        ).values_list('document_version_id', flat=True)

        # Filter to unsynced document versions only.
        unsynced_versions = DocumentVersion.objects.exclude(
            pk__in=synced_version_ids
        )

        total = unsynced_versions.count()
        success_count = 0
        error_count = 0

        self.stdout.write(
            'Found {} unsynced document version(s).'.format(total)
        )

        for document_version in unsynced_versions.iterator():
            self.stdout.write(
                'Syncing document version {} (document {})...'.format(
                    document_version.pk, document_version.document_id
                )
            )
            result = send_document_version_to_rag(
                document_version=document_version
            )
            if result:
                success_count += 1
            else:
                error_count += 1

        self.stdout.write('')
        self.stdout.write('Sync summary:')
        self.stdout.write('  Total:     {}'.format(total))
        self.stdout.write('  Success:   {}'.format(success_count))
        self.stdout.write('  Errors:    {}'.format(error_count))

        if error_count:
            self.stdout.write(self.style.WARNING(
                'Some document versions failed to sync.'
            ))
        else:
            self.stdout.write(self.style.SUCCESS(
                'All document versions synced successfully.'
            ))
