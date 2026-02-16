from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ('sources', '0028_auto_20210905_0558')
    ]

    operations = [
        migrations.RunPython(
            code=migrations.RunPython.noop,
            reverse_code=migrations.RunPython.noop
        )
    ]
