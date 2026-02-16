import logging

from mayan.celery import app

from .classes import StatisticType

logger = logging.getLogger(name=__name__)


@app.task(ignore_result=True)
def task_execute_statistic(slug):
    logger.debug('Executing')

    StatisticType.get(slug=slug).execute()

    logger.debug('Finshed')
