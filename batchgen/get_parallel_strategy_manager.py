from batchgen.models.deepseek.deepseekv3.Parallel_Strategy_Manager import DeepseekV3ParallelStrategyManager


def get_parallel_strategy_manager(model_name:str):
	if model_name.lower() in ["deepseek-ai/DeepSeek-R1", "deepseek-ai/DeepSeek-V3"]:
		return DeepseekV3ParallelStrategyManager
	else:
		raise ValueError(f"Unsupported model name: {model_name}")