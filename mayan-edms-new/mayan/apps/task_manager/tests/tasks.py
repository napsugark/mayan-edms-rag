from mayan.celery import app


@app.task(ignore_result=True)
def test_task():
    """Test task."""
