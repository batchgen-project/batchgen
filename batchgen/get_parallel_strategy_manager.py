from batchgen.models.deepseek.deepseekv3.Parallel_Strategy_Manager import DeepseekV3ParallelStrategyManager
from batchgen.models.openai.gpt_oss_120b.Parallel_Strategy_Manager import GptOssParallelStrategyManager
from batchgen.models.moonshotai.kimi_k25.Parallel_Strategy_Manager import KimiK25ParallelStrategyManager
from batchgen.config.model_name_utils import is_kimi_k25_backend_model
from batchgen.models.minimax.minimax_m25.Parallel_Strategy_Manager import MiniMaxM25ParallelStrategyManager


def get_parallel_strategy_manager(model_name:str):
	model_lower = model_name.lower()
	if "minimax" in model_lower or "minimax-m2.5" in model_lower:
		return MiniMaxM25ParallelStrategyManager
	elif is_kimi_k25_backend_model(model_name):
		return KimiK25ParallelStrategyManager
	elif model_lower in [
		"deepseek-ai/deepseek-r1",
		"deepseek-ai/deepseek-v3",
		"deepseek-ai/deepseek-v4-flash",
		"deepseek-ai/deepseek-v4-pro",
	] or "deepseek-v4" in model_lower:
		return DeepseekV3ParallelStrategyManager
	elif "gpt-oss-120b" in model_lower:
		return GptOssParallelStrategyManager
	elif "glm-5" in model_lower:
		from batchgen.models.glm.glm5.Parallel_Strategy_Manager import GLM5ParallelStrategyManager
		return GLM5ParallelStrategyManager
	else:
		raise ValueError(f"Unsupported model name: {model_name}")
