from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('storage', '0007_auto_20210218_0708')
    ]

    operations = [
        migrations.RemoveField(
            model_name='downloadfile',
            name='content_type',
        ),
        migrations.RemoveField(
            model_name='downloadfile',
            name='object_id',
        ),
        migrations.RemoveField(
            model_name='downloadfile',
            name='permission',
        ),
        migrations.AddField(
            model_name='downloadfile',
            name='user',
            field=models.ForeignKey(
                default=1, editable=False,
                on_delete=django.db.models.deletion.CASCADE,
                related_name='download_files', to='auth.user',
                verbose_name='User'
            ),
            preserve_default=False
        )
    ]
