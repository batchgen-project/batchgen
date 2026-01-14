from batchgen.models.deepseek.deepseekv3.Parallel_Strategy_Manager import DeepseekV3ParallelStrategyManager


def get_parallel_strategy_manager(model_name:str):
	model_lower = model_name.lower()
	if model_lower in ["deepseek-ai/deepseek-r1", "deepseek-ai/deepseek-v3"]:
		return DeepseekV3ParallelStrategyManager
	elif "gpt-oss" in model_lower or model_lower == "openai/gpt-oss-120b":
		return GptOssParallelStrategyManager
	else:
		raise ValueError(f"Unsupported model name: {model_name}")