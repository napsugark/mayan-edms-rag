import uuid

from django.core.files.storage import FileSystemStorage
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('documents', '0002_auto_20150608_1902')
    ]

    operations = [
        migrations.AlterField(
            model_name='document',
            name='uuid',
            field=models.CharField(
                default=str(
                    uuid.uuid4()
                ), editable=False, max_length=48
            ),
            preserve_default=True
        ),
        migrations.AlterField(
            model_name='documentversion',
            name='file',
            field=models.FileField(
                storage=FileSystemStorage(), upload_to=str(
                    uuid.uuid4()
                ), verbose_name='File'
            ),
            preserve_default=True
        )
    ]
