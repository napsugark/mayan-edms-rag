from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('checkouts', '0001_initial')
    ]

    operations = [
        migrations.AddField(
            model_name='documentcheckout',
            name='user',
            field=models.ForeignKey(
                blank=True, null=True, on_delete=models.CASCADE,
                to=settings.AUTH_USER_MODEL, verbose_name='User'
            ),
            preserve_default=True
        )
    ]
