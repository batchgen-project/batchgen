def get_parallel_strategy_manager(model_name:str):
	model_lower = model_name.lower()
	if model_lower in ["moonshotai/kimi-k2.5"]:
		from batchgen.models.moonshotai.kimi_k25.Parallel_Strategy_Manager import KimiK25ParallelStrategyManager
		return KimiK25ParallelStrategyManager
	elif model_lower in ["deepseek-ai/deepseek-r1", "deepseek-ai/deepseek-v3"]:
		from batchgen.models.deepseek.deepseekv3.Parallel_Strategy_Manager import DeepseekV3ParallelStrategyManager
		return DeepseekV3ParallelStrategyManager
	elif "gpt-oss-120b" in model_lower:
		from batchgen.models.openai.gpt_oss_120b.Parallel_Strategy_Manager import GptOssParallelStrategyManager
		return GptOssParallelStrategyManager
	elif "glm-5" in model_lower:
		from batchgen.models.glm.glm5.Parallel_Strategy_Manager import GLM5ParallelStrategyManager
		return GLM5ParallelStrategyManager
	else:
		raise ValueError(f"Unsupported model name: {model_name}")
