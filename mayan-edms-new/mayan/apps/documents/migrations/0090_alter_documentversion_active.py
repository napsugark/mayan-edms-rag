from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('documents', '0089_auto_20230829_0819')
    ]

    operations = [
        migrations.AlterField(
            model_name='documentversion', name='active',
            field=models.BooleanField(
                default=False, help_text='Determines the active version '
                'of the document.', verbose_name='Active'
            )
        )
    ]
