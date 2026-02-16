from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ('checkouts', '0010_auto_20200917_0643')
    ]

    operations = [
        migrations.DeleteModel(name='NewFileBlock')
    ]
