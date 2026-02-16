from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('mailer', '0008_migrate_to_credentials')
    ]

    operations = [
        migrations.AlterField(
            field=models.TextField(
                blank=True, help_text='JSON encoded data for the backend '
                'class.', verbose_name='Backend data'
            ), model_name='usermailer', name='backend_data'
        )
    ]
