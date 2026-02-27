from batchgen.models.deepseek.deepseekv3.deepseekv3_initializer import DeepseekV3Initializer
from batchgen.models.openai.gpt_oss_120b.gpt_oss_initializer import GptOssInitializer
from batchgen.models.moonshotai.kimi_k25.kimi_initializer import KimiK25Initializer
from batchgen.models.minimax.minimax_m25.minimax_m25_initializer import MiniMaxM25Initializer

def get_initializer(model_name:str):
	model_lower = model_name.lower()
	if "minimax" in model_lower or "minimax-m2.5" in model_lower:
		return MiniMaxM25Initializer
	elif model_lower in ["moonshotai/kimi-k2.5"]:
		return KimiK25Initializer
	elif model_lower in ["deepseek-ai/deepseek-r1", "deepseek-ai/deepseek-v3"]:
		return DeepseekV3Initializer
	elif "gpt-oss-120b" in model_lower:
		return GptOssInitializer
	else:
		raise ValueError(f"Unsupported model name: {model_name}")