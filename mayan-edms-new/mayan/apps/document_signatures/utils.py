import uuid


def upload_to(*args, **kwargs):
    return str(
        uuid.uuid4()
    )
