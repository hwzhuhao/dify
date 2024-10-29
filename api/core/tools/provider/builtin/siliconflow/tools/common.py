from typing import Any, Union

import requests

from core.tools.entities.tool_entities import ToolInvokeMessage


class ToolHelper:
    API_URL: str = "https://api.siliconflow.cn/v1/image/generations"

    @staticmethod
    def send_request(
        url: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> Union[ToolInvokeMessage, list[ToolInvokeMessage]]:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            return ToolInvokeMessage.create_text_message(f"Got Error Response: {response.text}")

        res = response.json()
        result = [ToolInvokeMessage.create_json_message(res)]
        for image in res.get("images", []):
            result.append(ToolInvokeMessage.create_image_message(image=image.get("url"), save_as="IMAGE"))
        return result

    @staticmethod
    def get_headers(api_key: str) -> dict[str, str]:
        return {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}",
        }
