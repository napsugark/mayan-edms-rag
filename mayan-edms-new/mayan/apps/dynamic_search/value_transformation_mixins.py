import datetime


class ValueTransformationTimezoneMixin:
    def _execute(self):
        value = self._execute_()

        if value is not None:
            pass
            try:
                tzinfo = value.tzinfo
            except AttributeError:
                return value
            else:
                if not tzinfo:
                    value = value.replace(tzinfo=datetime.timezone.utc)

        return value
