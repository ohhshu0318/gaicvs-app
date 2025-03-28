import asyncio
import json
from datetime import datetime, timezone
from uuid import uuid4

import httpx

from classes.logging import logger
from common.const import ACCGROUP, APITOKEN, CMDBID, COSTCENTER, FLORENCE_URL

limit = asyncio.Semaphore(20)
http_client = httpx.AsyncClient()


class FlorenceOperation:
    def __init__(
        self, cmdbid: str, accgroup: str, costcenter: str, url: str, token: str
    ) -> None:
        self.cmdbid = cmdbid
        self.accgroup = accgroup
        self.costcenter = costcenter
        self.url = url
        self.token = token

    def chat(self, prompts: list[str]) -> list[dict]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(
                self.task_pool(
                    requestbody_list=[
                        {
                            "model": "gpt-4-32k",
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False,
                            "temperature": 0,
                            "n": 1,
                            "max_tokens": 500,
                        }
                        for prompt in prompts
                    ],
                    endpoint="process",
                )
            )
            loop.close()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(
                self.task_pool(
                    requestbody_list=[
                        {
                            "model": "gpt-4-32k",
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False,
                            "temperature": 0,
                            "n": 1,
                            "max_tokens": 500,
                        }
                        for prompt in prompts
                    ],
                    endpoint="process",
                )
            )
            loop.close()
        return results

    def embed(self, texts: list[str]) -> list[dict]:
        return asyncio.run(
            self.task_pool(
                requestbody_list=[
                    {
                        "model": "amazon.titan-embed-text-v1",
                        "accept": "application/json",
                        "contentType": "application/json",
                        "body": {"inputText": text},
                    }
                    for text in texts
                ],
                endpoint="embed",
            )
        )

    async def make_one_requests(
        self, requestbody: dict, endpoint: str, **kwargs
    ) -> dict:
        async with limit:
            if endpoint not in ("process", "embed"):
                raise ValueError("endpoint must be 'process' or 'embed'")
            if (
                endpoint == "process"
                and not requestbody["messages"][0]["content"]
            ):
                return {"status_code": 200, "data": "", "text": ""}
            payload = json.dumps(
                {
                    "cmdbid": self.cmdbid,
                    "accgroup": self.accgroup,
                    "businessunit": "POJ",
                    "costcenter": self.costcenter,
                    "apicontext": (
                        "chatcompletions"
                        if endpoint == "process"
                        else "embeddings"
                    ),
                    "provider": (
                        "azure-openai"
                        if endpoint == "process"
                        else "amazon-bedrock"
                    ),
                    "requestid": str(uuid4()),
                    "requestdatetime": (
                        datetime.now(tz=timezone.utc).strftime(
                            "%Y-%m-%d %H:%M:%S%z"
                        )
                    ),
                    "requestbody": requestbody,
                }
            )
            headers = {
                "cmdbid": self.cmdbid,
                "apitoken": self.token,
                "Content-Type": "application/json",
            }
            response = await http_client.post(
                url=f"{self.url}/{endpoint}",
                headers=headers,
                data=payload,
                timeout=None,
            )
            if limit.locked():
                await asyncio.sleep(1)
            if response.status_code == 200:
                data = (
                    response.json()["responsebody"]["choices"][0]["message"][
                        "content"
                    ]
                    if endpoint == "process"
                    else response.json()["responsebody"]["embedding"]
                )
                return {
                    "status_code": response.status_code,
                    "data": data,
                    "text": "",
                }
            elif response.status_code == 500:
                logger.error(
                    f"Florenceへリクエストした際に、エラーが発生した。\n  response code:{response.status_code}\n  response text:{response.text}"
                )
                await asyncio.sleep(1)
                response = await http_client.post(
                    url=f"{self.url}/{endpoint}",
                    headers=headers,
                    data=payload,
                    timeout=None,
                )
                if response.status_code == 200:
                    data = (
                        response.json()["responsebody"]["choices"][0][
                            "message"
                        ]["content"]
                        if endpoint == "process"
                        else response.json()["responsebody"]["embedding"]
                    )
                else:
                    data = None
                return {
                    "status_code": response.status_code,
                    "data": data,
                    "text": response.text,
                }
            else:
                logger.error(
                    f"Florenceへリクエストした際に、エラーが発生した。\n  response code:{response.status_code}\n  response text:{response.text}"
                )
                return {
                    "status_code": response.status_code,
                    "data": None,
                    "text": response.text,
                }

    async def task_pool(self, requestbody_list, endpoint):
        tasks = []
        for requestbody in requestbody_list:
            task = asyncio.create_task(
                self.make_one_requests(requestbody, endpoint)
            )
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        return results


florence_operator = FlorenceOperation(
    CMDBID, ACCGROUP, COSTCENTER, FLORENCE_URL, APITOKEN
)
