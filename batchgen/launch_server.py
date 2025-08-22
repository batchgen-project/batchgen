"""
	Launch the server.
	Node 0 init the process group in the multi-node case.
	1. Config the placement of the model.
	2. Launch Local Parameter Server. (DP-Attn, DP-SharedExpert, EP-RoutedExpert)
	3. Spawn Local Worker Processes(With its own pinned KV-Cache, bind to device-numa)
	4. Init Dist Process Group. 
	5. Master Node (Node 0) Start Dispatcher Processes. 
		Listen to the incoming requests from Client and dispatch prompts to DP-Workers.
"""

def parse_args():
	import argparse
	parser = argparse.ArgumentParser(description="Launch BatchGen Server")
	parser.add_argument(
		"--model", 
		type=str, 
		help="We follow the Huggingface naming such as 'deepseek-ai/DeepSeek-R1'."
	)
	
	parser.add_argument(
		"--checkpoint-path", 
		type=str, 
		help="Path to the checkpoint directory. \
			E.g.'/root/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1'."
	)

	parser.add_argument(
		"--host-mem-usage",
		type=int,
		help="Size of the host memory will be occupied per node by BatchGen. In GB. \
			For example, for a 1TB host DRAM, 900GB is recommended to prevent host OOM.  \
			The rest space is often reserved for the OS and docker. "
	)

	parser.add_argument(
		"--num-device",
		type=int,
		help="Total number of GPUs to use. E.g. 16 for two nodes of 8 GPUs each."
	)

	parser.add_argument(
		"--dist-init-addr",
		type=str,
		help="The address of the master node and the port number. \
			For torch dist process group init. \
			E.g. '10.0.0.8:12345'."
	)

	parser.add_argument(
		"--num-nodes",
		type=int,
		help="Total number of nodes. E.g. 2 for two nodes."
	)

	parser.add_argument(
		"--node-rank",
		type=int,
		help="The rank of the current node. \
			E.g. 0 for the first node, 1 for the second node."
	)

	parser.add_argument(
		"--host",
		type=str,
		default="0.0.0.0",
		help="By default, the server will listen on all interfaces. \
			You can specify a specific interface to listen on."
	)

	parser.add_argument(
		"--port",
		type=int,
		default=13355,
		help="The port number to listen on. \
			You can specify a different port when there is a port conflict."
	)

	return parser.parse_args()



	


class MoegenServer:
	def __init__(self, args):
		self.args = args
