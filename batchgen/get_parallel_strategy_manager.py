KIMI_K25_BACKEND_NAME_PATTERNS = (
	"moonshotai/kimi-k2.5",
	"moonshotai/kimi-k2.6",
	"kimi-k2.5",
	"kimi_k2.5",
	"kimi-k25",
	"kimi_k25",
	"kimi-k2.6",
	"kimi_k2.6",
	"kimi-k26",
	"kimi_k26",
)


def _is_kimi_k25_backend_model(model_name: str) -> bool:
	model_lower = model_name.strip().lower()
	return any(pattern in model_lower for pattern in KIMI_K25_BACKEND_NAME_PATTERNS)


def get_parallel_strategy_manager(model_name:str):
	model_lower = model_name.lower()
	if "minimax" in model_lower or "minimax-m2.5" in model_lower:
		from batchgen.models.minimax.minimax_m25.Parallel_Strategy_Manager import MiniMaxM25ParallelStrategyManager
		return MiniMaxM25ParallelStrategyManager
	elif _is_kimi_k25_backend_model(model_name):
		from batchgen.models.moonshotai.kimi_k25.Parallel_Strategy_Manager import KimiK25ParallelStrategyManager
		return KimiK25ParallelStrategyManager
	elif "deepseek-v4" in model_lower:
		from batchgen.models.deepseek.deepseekv4_flash.Parallel_Strategy_Manager import DeepSeekV4FlashParallelStrategyManager
		return DeepSeekV4FlashParallelStrategyManager
	elif model_lower in [
		"deepseek-ai/deepseek-r1",
		"deepseek-ai/deepseek-v3",
	]:
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
