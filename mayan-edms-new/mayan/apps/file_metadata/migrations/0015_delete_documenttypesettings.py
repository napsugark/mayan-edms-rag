from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ('file_metadata', '0014_document_type_driver_populate')
    ]

    operations = [
        migrations.DeleteModel(name='DocumentTypeSettings')
    ]
