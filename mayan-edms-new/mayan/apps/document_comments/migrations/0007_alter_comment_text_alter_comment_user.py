from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('document_comments', '0006_auto_20210130_0658')
    ]

    operations = [
        migrations.AlterField(
            field=models.TextField(
                help_text='Actual text content of the comment.',
                verbose_name='Text'
            ), model_name='comment', name='text'
        ),
        migrations.AlterField(
            field=models.ForeignKey(
                editable=False, help_text='The user account that made the '
                'comment.', on_delete=django.db.models.deletion.CASCADE,
                related_name='comments', to=settings.AUTH_USER_MODEL,
                verbose_name='User'
            ), model_name='comment', name='user'
        )
    ]
