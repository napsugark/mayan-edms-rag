from django.db import models

from .literals import DEFAULT_CREATE_BULK_BATCH_SIZE


class ManagerMinixCreateBulk(models.Manager):
    create_bulk_batch_size = DEFAULT_CREATE_BULK_BATCH_SIZE

    def create_bulk(self):
        batch = []
        count = 0

        try:
            while True:
                kwargs = (yield)
                model_instance = self.model(**kwargs)
                batch.append(model_instance)
                count += 1

                if count >= self.create_bulk_batch_size:
                    count = 0

                    self.bulk_create(
                        batch_size=self.create_bulk_batch_size, objs=batch
                    )
                    batch = []
        except GeneratorExit:
            self.bulk_create(
                batch_size=self.create_bulk_batch_size, objs=batch
            )
