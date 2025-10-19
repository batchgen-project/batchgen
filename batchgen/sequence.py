"""
	Data structure for storing and updating query information.
"""
from enum import IntEnum
class SequenceStatus(IntEnum):
	QUEUEING = 0
	IN_PREFILL = 1
	PREFILLED = 2
	IN_DECODE = 3
	ON_HOLD = 4
	COMPLETED = 5

class SequenceEntry:
	__slots__ = ('uuid', 'prompt_length', 'max_decode_length', 
			  	'status', 'decoded_length', 'current_context_length',
				'input_ids')

	VALID_TRANSITIONS = {
			SequenceStatus.QUEUEING: {SequenceStatus.IN_PREFILL},
			SequenceStatus.IN_PREFILL: {SequenceStatus.PREFILLED}, # Currently the prefill would not be interrupted.
			SequenceStatus.PREFILLED: {SequenceStatus.IN_DECODE, SequenceStatus.ON_HOLD},
			SequenceStatus.IN_DECODE: {SequenceStatus.ON_HOLD, SequenceStatus.COMPLETED},
			SequenceStatus.ON_HOLD: {SequenceStatus.IN_DECODE},
			SequenceStatus.COMPLETED: set(),  # No transitions allowed from COMPLETED
	}	

	def __init__(self, uuid:str, prompt_length:int, max_decode_length:int):
		self.uuid = uuid
		self.prompt_length = prompt_length
		self.max_decode_length = max_decode_length
		self.status = SequenceStatus.QUEUEING
		self.decoded_length = 0
		self.current_context_length = prompt_length
		self.input_ids = None  # To be set when the query is processed
	
	def status_transition(self, new_status:SequenceStatus):
		if new_status in self.VALID_TRANSITIONS[self.status]:
			self.status = new_status
		else:
			raise ValueError(f"Invalid status transition from {self.status} to {new_status}")
	

class SequenceBatch:
	__slots__ = ('sequences')
	def __init__(self):
		self.sequences = {} # uuid -> SequenceEntry
	def add_sequence(self, sequence:SequenceEntry):
		self.sequences[sequence.uuid] = sequence
	def get_sequence(self, uuid:str) -> SequenceEntry:
		return self.sequences.get(uuid, None)
	def update_status(self, uuid:str, new_status:SequenceStatus):
		if uuid in self.sequences:
			self.sequences[uuid].status = new_status
		else:
			raise KeyError(f"Sequence with UUID {uuid} not found.")
	def create_view(self, uuids: list[str]) -> 'SequenceBatch':
		"""
		Create a view batch containing only specified UUIDs.
		Returns references to the same sequence objects.
		
		Args:
			uuids: List of UUIDs to include in the view
		
		Returns:
			New SequenceBatch with references to specified sequences
		"""
		view = SequenceBatch()
		for uuid in uuids:
			if uuid in self.sequences:
				view.sequences[uuid] = self.sequences[uuid]  # Reference!
		return view


