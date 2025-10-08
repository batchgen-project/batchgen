from batchgen.models.deepseek.deepseekv3.deepseekv3_initializer import DeepseekV3Initializer

def get_initializer(model_name:str):
	if model_name.lower() in ["deepseek-ai/deepseek-r1", "deepseek-ai/deepseek-v3"]:
		return DeepseekV3Initializer
	else:
		raise ValueError(f"Unsupported model name: {model_name}")