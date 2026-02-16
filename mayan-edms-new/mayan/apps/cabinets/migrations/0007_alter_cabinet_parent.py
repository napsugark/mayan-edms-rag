from django.db import migrations
import django.db.models.deletion

import mptt.fields


class Migration(migrations.Migration):
    dependencies = [
        ('cabinets', '0006_auto_20210525_0604')
    ]

    operations = [
        migrations.AlterField(
            model_name='cabinet',
            name='parent',
            field=mptt.fields.TreeForeignKey(
                blank=True, null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name='children', to='cabinets.cabinet',
                verbose_name='Parent'
            )
        )
    ]
