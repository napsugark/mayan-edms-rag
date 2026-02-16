from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ('metadata', '0019_alter_documentmetadata_value')
    ]

    operations = [
        migrations.CreateModel(
            bases=('metadata.documentmetadata',), fields=[],
            name='DocumentMetadataSearchResult', options={
                'proxy': True,
                'indexes': [],
                'constraints': []
            }
        )
    ]
