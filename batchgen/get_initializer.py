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

def get_initializer(model_name:str):
	model_lower = model_name.lower()
	if "minimax" in model_lower or "minimax-m2.5" in model_lower:
		from batchgen.models.minimax.minimax_m25.minimax_m25_initializer import MiniMaxM25Initializer
		return MiniMaxM25Initializer
	elif "kimi-linear" in model_lower or "kimi-k3" in model_lower:
		from batchgen.models.moonshotai.kimi_linear.kimi_initializer import KimiLinearInitializer
		return KimiLinearInitializer
	elif _is_kimi_k25_backend_model(model_name):
		from batchgen.models.moonshotai.kimi_k25.kimi_initializer import KimiK25Initializer
		return KimiK25Initializer
	elif "deepseek-v4" in model_lower:
		from batchgen.models.deepseek.deepseekv4_flash.deepseekv4_flash_initializer import DeepSeekV4FlashInitializer
		return DeepSeekV4FlashInitializer
	elif model_lower in [
		"deepseek-ai/deepseek-r1",
		"deepseek-ai/deepseek-v3",
	]:
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
