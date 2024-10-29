from typing import Any, Union

from core.tools.entities.tool_entities import ToolInvokeMessage
from core.tools.provider.builtin.siliconflow.tools.common import ToolHelper
from core.tools.tool.builtin_tool import BuiltinTool


class StableDiffusionTool(BuiltinTool):
    @classmethod
    def get_model_map(cls) -> dict[str, str]:
        return {
            "sd_3": "stabilityai/stable-diffusion-3-medium",
            "sd_xl": "stabilityai/stable-diffusion-xl-base-1.0",
            "sd_3.5_large": "stabilityai/stable-diffusion-3-5-large",
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
            "negative_prompt": tool_parameters.get("negative_prompt", ""),
            "image_size": tool_parameters.get("image_size", "1024x1024"),
            "batch_size": tool_parameters.get("batch_size", 1),
            "seed": tool_parameters.get("seed"),
            "guidance_scale": tool_parameters.get("guidance_scale", 7.5),
            "num_inference_steps": tool_parameters.get("num_inference_steps", 20),
        }
