import os

import requests


class ECSOperation:
    def __init__(self,) -> None:
        pass

    def get_task_id(self) -> str:
        r = requests.get(
            f"{os.environ.get('ECS_CONTAINER_METADATA_URI_V4')}/task"
            )
        return str(r.json()['TaskARN']).split("/")[0]


ecs_operator = ECSOperation
