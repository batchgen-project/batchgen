from batchgen.models.deepseek.deepseekv3.deepseekv3_initializer import DeepseekV3Initializer
from batchgen.models.openai.gpt_oss_120b.gpt_oss_initializer import GptOssInitializer
from batchgen.models.moonshotai.kimi_k25.kimi_initializer import KimiK25Initializer
from batchgen.config.model_name_utils import is_kimi_k25_backend_model
from batchgen.models.minimax.minimax_m25.minimax_m25_initializer import MiniMaxM25Initializer

def get_initializer(model_name:str):
	model_lower = model_name.lower()
	if "minimax" in model_lower or "minimax-m2.5" in model_lower:
		return MiniMaxM25Initializer
	elif is_kimi_k25_backend_model(model_name):
		return KimiK25Initializer
	elif model_lower in ["deepseek-ai/deepseek-r1", "deepseek-ai/deepseek-v3"]:
		return DeepseekV3Initializer
	elif "gpt-oss-120b" in model_lower:
		return GptOssInitializer
	elif "glm-5" in model_lower:
		from batchgen.models.glm.glm5.glm5_initializer import GLM5Initializer
		return GLM5Initializer
	else:
		raise ValueError(f"Unsupported model name: {model_name}")
