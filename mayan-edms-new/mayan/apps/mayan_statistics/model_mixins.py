import json


class StatisticResultBusinessLogicMixin:
    def get_data(self):
        return json.loads(s=self.serialize_data)

    def store_data(self, data):
        self.serialize_data = json.dumps(obj=data)
        self.save()
