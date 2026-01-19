from batchgen.models.deepseek.deepseekv3.deepseekv3_initializer import DeepseekV3Initializer
from batchgen.models.gpt_oss.gpt_oss_initializer import GptOssInitializer

def get_initializer(model_name:str):
	model_lower = model_name.lower()
	if model_lower in ["deepseek-ai/deepseek-r1", "deepseek-ai/deepseek-v3"]:
		return DeepseekV3Initializer
	elif "gpt-oss-120b" in model_lower:
		return GptOssInitializer
	else:
		raise ValueError(f"Unsupported model name: {model_name}")