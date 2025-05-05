import argparse
import logging
import socket
SUPPORTED_MODELS = ['deepseek-ai/DeepSeek-R1']


# def validate_host_kv_capacity(value):
# 	ivalue = int(value)
# 	# Get the free physical memory in GB

def validate_server_host(value):
	if value.lower() == 'localhost':
		return value.lower()
	
	# Check if value is an IP address
	try:
		socket.inet_aton(value)
		# It's a valid IP address
		return value
	except socket.error:
		# Not an IP address, try to resolve as hostname
		try:
			socket.gethostbyname(value)
			return value
		except socket.gaierror:
			logging.warning(f"Could not resolve host: {value}. Server might not be reachable.")
			# Still return the value to allow use of unresolvable hosts (for testing)
			return value	

def validate_server_port(value):
    """Validate that the port is in a valid range."""
    ivalue = int(value)
    if ivalue < 1024 or ivalue > 65535:
        raise argparse.ArgumentTypeError(f"Port should be between 1024 and 65535, got {ivalue}")
    return ivalue



def server_argparser():
	parser = argparse.ArgumentParser(description='Server Configuration Arguments')
	parser.add_argument(
		'--model_name',
		type=str,
		required=True,
		choices=SUPPORTED_MODELS, # TODO: add supported model list.
		help=f'Model name. Must be one of: {", ".join(SUPPORTED_MODELS)}'
	)

	parser.add_argument(
		'--host_kv_capacity',
		type=int,
		required=True,
		help='Size of the host memory that would be reserved (in GB).'
	)

	parser.add_argument(
		'--ep',
		type=bool,
		help='Whether expert parallelism is turned on. Currently only supports 2, 4, or 8 devices within a node.'
	)

	parser.add_argument(
		'--server_host',
		type=validate_server_host,
		default='localhost',
		help='Server host address. Default is localhost.'
	)	

	parser.add_argument(
		'--server_port',
		type=validate_server_port,
		default=9090,
		help='Server port number. Default is 9090.'
	)

	args = parser.parse_args()
	logging.info(f"Server arguments: {args}")

	return args