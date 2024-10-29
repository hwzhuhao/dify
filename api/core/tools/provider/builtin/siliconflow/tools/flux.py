from typing import Any, Union

from core.tools.entities.tool_entities import ToolInvokeMessage
from core.tools.provider.builtin.siliconflow.tools.common import ToolHelper
from core.tools.tool.builtin_tool import BuiltinTool


class FluxTool(BuiltinTool):
    @classmethod
    def get_model_map(cls) -> dict[str, str]:
        return {
            "schnell": "black-forest-labs/FLUX.1-schnell",
            "dev": "black-forest-labs/FLUX.1-dev",
        }

    def _invoke(
        self, user_id: str, tool_parameters: dict[str, Any]
    ) -> Union[ToolInvokeMessage, list[ToolInvokeMessage]]:
        headers = ToolHelper.get_headers(self.runtime.credentials["siliconFlow_api_key"])
        model = self.get_model_map().get(tool_parameters.get("model"), list(self.get_model_map().values())[0])
        payload = self.build_payload(tool_parameters, model)
        return ToolHelper.send_request(ToolHelper.API_URL, payload, headers)

    @classmethod
    def build_payload(cls, tool_parameters: dict[str, Any], model: str) -> dict[str, Any]:
        return {
            "model": model,
            "prompt": tool_parameters.get("prompt"),
            "image_size": tool_parameters.get("image_size", "1024x1024"),
            "seed": tool_parameters.get("seed"),
            "num_inference_steps": tool_parameters.get("num_inference_steps", 20),
        }
