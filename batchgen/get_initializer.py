def get_initializer(model_name:str):
	model_lower = model_name.lower()
	if model_lower in ["moonshotai/kimi-k2.5"]:
		from batchgen.models.moonshotai.kimi_k25.kimi_initializer import KimiK25Initializer
		return KimiK25Initializer
	elif model_lower in ["deepseek-ai/deepseek-r1", "deepseek-ai/deepseek-v3"]:
		from batchgen.models.deepseek.deepseekv3.deepseekv3_initializer import DeepseekV3Initializer
		return DeepseekV3Initializer
	elif "gpt-oss-120b" in model_lower:
		from batchgen.models.openai.gpt_oss_120b.gpt_oss_initializer import GptOssInitializer
		return GptOssInitializer
	elif "glm-5" in model_lower:
		from batchgen.models.glm.glm5.glm5_initializer import GLM5Initializer
		return GLM5Initializer
	else:
		raise ValueError(f"Unsupported model name: {model_name}")
