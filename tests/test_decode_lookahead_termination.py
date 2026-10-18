"""One-step lookahead must not move a termination boundary or a KV write.

Lookahead defers step N's host bookkeeping into step N+1, so the host does not
know a row has terminated until one step later and launches one more forward
for it. Two things must survive that:

* **Termination exactness.** A request asking for ``max_tokens = k`` receives
  exactly ``k`` tokens, and a row stopping on a stop token emits nothing after
  it. The extra computed token is discarded, never appended.
* **Page coverage.** The extra forward uses the context the row would have had
  after its own final bookkeeping -- the same value the synchronous path keeps
  passing for a retired row until the next boundary -- so the per-row page
  requirement is unchanged. The GPU write index and the host mirror's write
  index are both ``cache_seqlens - 1``, so the requirement is
  ``cache_seqlens <= pages * PAGE_SIZE``, and the test checks it on both
  sides for both ``(prompt + max_tokens) % 64`` classes. The aligned class is
  the dangerous one: there the last page is exactly full.

The bookkeeping under test is the real ``_apply_pending_decode_tokens``,
lifted from the worker source (``batchgen_worker`` cannot be imported on a
CPU-only machine), driven over a scripted token stream with the real
completion predicate and the real ``effective_cache_seqlens``. Nothing in the
termination rules is re-implemented here.
"""
import ast
import math
from pathlib import Path

import pytest

from batchgen.worker.completion import CompletionContext, CompletionHandler
from batchgen.worker.decode_lookahead import (
    DecodeLookaheadState,
    effective_cache_seqlens,
)
from tests.worker_method_source import load_worker_methods

PAGE_SIZE = 64
EXTENSION_GPU_PAGE_BUFFER = 4
MODEL_CONTEXT_LENGTH = 1 << 30  # not the limit under test


# --------------------------------------------------------------------------
# fakes: only what the lifted method and the completion predicate touch
# --------------------------------------------------------------------------
class FakeTokenView:
	"""Stands in for a ``[n, 1]`` pinned tensor slice."""

	def __init__(self, rows):
		self._rows = list(rows)

	def store(self, rows):
		self._rows = list(rows)

	def __getitem__(self, key):
		return FakeTokenView(self._rows[key])

	def reshape(self, _shape):
		return self

	def tolist(self):
		return list(self._rows)


class FakeEvent:
	def __init__(self):
		self.syncs = 0

	def synchronize(self):
		self.syncs += 1


class FakeSeq:
	def __init__(self, uuid, prompt, max_tokens, stop_tokens=()):
		self.uuid = uuid
		self.global_idx = int(uuid)
		self.original_prompt_length = prompt
		self.prompt_length = prompt
		self.max_decode_length = max_tokens
		self.stop_tokens = set(stop_tokens)
		# Post-prefill state: the prompt is in the cache, nothing decoded.
		self.decoded_length = 0
		self.current_context_length = prompt
		self.eos_reached = False
		self._rep_detected = False
		self._rep_last_token = None
		self._rep_count = 0
		self._lifespan_log = []

	@property
	def kv_token_budget(self):
		return self.original_prompt_length + self.max_decode_length

	def log_event(self, *args, **kwargs):
		pass


class StubWorker:
	rank = 0

	def __init__(self):
		self.appended = []  # (uuid, token) in append order

	def _append_trajectory_tokens(self, sequences, token_ids):
		assert len(sequences) == len(token_ids)
		for seq, token in zip(sequences, token_ids):
			self.appended.append((seq.uuid, token))

	def _should_stop_at_eos(self, token_id, seq):
		return token_id in seq.stop_tokens

	def _trajectory_decoded_tokens(self, seq):
		return [tok for uuid, tok in self.appended if uuid == seq.uuid]

	def tokens_of(self, seq):
		return self._trajectory_decoded_tokens(seq)


class _Lifespan:
	@staticmethod
	def dump_lifespan(*args, **kwargs):
		pass


class _SeqEvent:
	REPETITION = "REPETITION"


_APPLY = load_worker_methods(
	["_apply_pending_decode_tokens"],
	{
		"logging": __import__("logging"),
		"time": __import__("time"),
		"BATCHGEN_MULTI_BATCH_DIAG": False,
		"REP_DETECTION": False,
		"_repetition_check_enabled": lambda seq: False,
		"_check_repeating_pattern": lambda tokens, dl: False,
		"lifespan": _Lifespan,
		"SeqEvent": _SeqEvent,
		"DecodeLookaheadState": DecodeLookaheadState,
		"Optional": __import__("typing").Optional,
		"List": __import__("typing").List,
	},
)["_apply_pending_decode_tokens"]


# --------------------------------------------------------------------------
# the driver: the step law around the real bookkeeping
# --------------------------------------------------------------------------
def _gpu_pages(seq):
	"""``SequenceEntry.get_gpu_pages_for_two_page_buffer`` for a decode row."""
	buffer_tokens = min(
		seq.current_context_length + EXTENSION_GPU_PAGE_BUFFER * PAGE_SIZE,
		seq.kv_token_budget,
	)
	return math.ceil(buffer_tokens / PAGE_SIZE)


def _validate_page_capacity(pages, required_tokens, where):
	"""The host mirror's own bound check (``ValidatePageCapacity``)."""
	if required_tokens > pages * PAGE_SIZE:
		raise AssertionError(
			f"{where}: page table too small for requested tokens -- "
			f"need {required_tokens}, have {pages} pages x {PAGE_SIZE}"
		)


def run_decode(seqs, token_script, *, lookahead, decision_interval):
	"""Return (worker, per-step cache_seqlens, max cache_seqlens per row)."""
	worker = StubWorker()
	state = DecodeLookaheadState()
	pinned = [FakeTokenView([]), FakeTokenView([])]
	events = [FakeEvent(), FakeEvent()]
	ctx = CompletionContext(
		eos_token_ids=frozenset(),
		model_context_length=MODEL_CONTEXT_LENGTH,
		rank=0,
	)
	rows = list(range(len(seqs)))
	# Pages are (re)planned at a boundary only, exactly like the real loop.
	pages = [_gpu_pages(s) for s in seqs]
	steps = []
	last_boundary = 0

	def apply_pending(iteration):
		return _APPLY(worker, state, pinned, events, iteration)

	for step, tokens in enumerate(token_script, start=1):
		boundary_due = step - last_boundary > decision_interval
		if boundary_due:
			# A boundary flushes the pipeline, then replans pages.
			apply_pending(step)
			pages = [_gpu_pages(s) for s in seqs]
			last_boundary = step

		offsets = state.context_offsets(rows)
		cache_seqlens = effective_cache_seqlens(
			[s.current_context_length for s in seqs], offsets
		)
		for i, seq in enumerate(seqs):
			# GPU side: FlashAttention writes this token at cache_seqlens - 1,
			# so the row needs cache_seqlens tokens of page coverage.
			assert cache_seqlens[i] <= pages[i] * PAGE_SIZE, (
				f"step {step} row {seq.uuid}: cache_seqlens={cache_seqlens[i]} "
				f"exceeds {pages[i]} pages x {PAGE_SIZE}"
			)
			# Host side: the mirror writes at the same index and validates
			# start_token + 1 against the same page table.
			_validate_page_capacity(
				pages[i], (cache_seqlens[i] - 1) + 1,
				f"step {step} row {seq.uuid}",
			)
		steps.append(tuple(cache_seqlens))

		slot = step % 2
		pinned[slot].store(tokens)
		if lookahead:
			apply_pending(step)
		state.arm(
			rows,
			seqs,
			[
				not CompletionHandler.is_sequence_completed(ctx, s)
				for s in seqs
			],
			slot,
		)
		if not lookahead:
			apply_pending(step)

	apply_pending(len(token_script) + 1)
	assert not state.is_armed
	return worker, steps


def _compare_modes(seqs_factory, token_script, decision_interval=4):
	sync_seqs = seqs_factory()
	la_seqs = seqs_factory()
	sync_worker, sync_steps = run_decode(
		sync_seqs, token_script, lookahead=False,
		decision_interval=decision_interval,
	)
	la_worker, la_steps = run_decode(
		la_seqs, token_script, lookahead=True,
		decision_interval=decision_interval,
	)
	# Identical device inputs every step => identical compute => the sampled
	# tokens stay bit-identical. This is the G4 claim, checked on the vector
	# the forward actually consumes.
	assert la_steps == sync_steps
	for a, b in zip(sync_seqs, la_seqs):
		assert la_worker.tokens_of(b) == sync_worker.tokens_of(a)
		assert (b.decoded_length, b.current_context_length, b.eos_reached) == (
			a.decoded_length, a.current_context_length, a.eos_reached
		)
	return sync_seqs, la_seqs, sync_worker, la_worker


# --------------------------------------------------------------------------
# termination boundaries
# --------------------------------------------------------------------------
@pytest.mark.parametrize("prompt", [448, 450], ids=["budget-aligned", "budget-unaligned"])
def test_max_tokens_exact_is_not_overrun(prompt):
	"""A row ending exactly at max_tokens gets exactly max_tokens tokens.

	``prompt=448`` makes ``budget = 448 + 64 = 512 = 8 x 64``: the last page is
	exactly full, which is the case where a single extra token of context
	would have no page to live in.
	"""
	max_tokens = 64
	if prompt == 448:
		assert (prompt + max_tokens) % PAGE_SIZE == 0
	else:
		assert (prompt + max_tokens) % PAGE_SIZE != 0

	# More steps than the row can accept, so the lookahead definitely runs a
	# forward for an already-terminated row.
	script = [[1000 + i] for i in range(max_tokens + 5)]
	sync_seqs, la_seqs, _, la_worker = _compare_modes(
		lambda: [FakeSeq("1", prompt, max_tokens)], script
	)
	(seq,) = la_seqs
	assert seq.decoded_length == max_tokens
	assert seq.eos_reached
	assert la_worker.tokens_of(seq) == [1000 + i for i in range(max_tokens)]
	assert seq.current_context_length == seq.kv_token_budget


def test_stop_token_emits_nothing_after_the_stop():
	stop = 7
	script = [[1], [2], [stop], [4], [5], [6], [7], [8]]
	_, la_seqs, _, la_worker = _compare_modes(
		lambda: [FakeSeq("1", 128, 32, stop_tokens=(stop,))], script
	)
	(seq,) = la_seqs
	assert la_worker.tokens_of(seq) == [1, 2, stop]
	assert seq.decoded_length == 3
	assert seq.eos_reached


def test_rows_terminating_at_different_steps_keep_their_own_boundaries():
	stop = 99
	script = [
		[1, 11, 21],
		[2, stop, 22],
		[3, 12, 23],
		[stop, 13, 24],
		[4, 14, 25],
		[5, 15, 26],
		[6, 16, 27],
	]
	_, la_seqs, _, la_worker = _compare_modes(
		lambda: [
			FakeSeq("1", 128, 16, stop_tokens=(stop,)),
			FakeSeq("2", 128, 16, stop_tokens=(stop,)),
			FakeSeq("3", 128, 4),
		],
		script,
		decision_interval=3,
	)
	first, second, third = la_seqs
	assert la_worker.tokens_of(first) == [1, 2, 3, stop]
	assert la_worker.tokens_of(second) == [11, stop]
	# Row 3 is length-capped at 4 and must get exactly 4 tokens.
	assert la_worker.tokens_of(third) == [21, 22, 23, 24]
	assert third.decoded_length == 4


def test_a_retired_rows_context_is_frozen_not_advanced():
	"""The overshoot must not accumulate: one extra forward, not one per step."""
	stop = 5
	script = [[1], [stop]] + [[9]] * 6
	seqs = [FakeSeq("1", 128, 32, stop_tokens=(stop,))]
	_, steps = run_decode(seqs, script, lookahead=True, decision_interval=4)
	(seq,) = seqs
	assert seq.decoded_length == 2
	# cache_seqlens climbs to the post-termination context and then stops.
	per_row = [s[0] for s in steps]
	assert per_row[:4] == [128, 129, 130, 130]
	assert per_row[3:] == [130] * (len(per_row) - 3)


def test_the_terminal_context_has_exactly_zero_page_slack_when_aligned():
	"""Why the lag must be exactly one step, and the margin that proves it.

	At its terminal context a row's page coverage is
	``ceil(budget / 64) * 64``, which equals ``budget`` exactly when
	``budget % 64 == 0``. The terminal context equals ``budget``, so the
	coverage is used to the last token: one token MORE than the terminal
	context has nowhere to live. One-step lookahead never asks for it -- the
	extra forward reuses the terminal context, and the token it produces is
	discarded -- but a two-step lag, or a lookahead that advanced the context
	of a retired row, would, and it would have to reserve pages first.
	"""
	seq = FakeSeq("1", 448, 64)
	assert seq.kv_token_budget % PAGE_SIZE == 0
	seq.decoded_length = seq.max_decode_length
	seq.current_context_length = seq.kv_token_budget
	coverage = _gpu_pages(seq) * PAGE_SIZE
	assert coverage == seq.kv_token_budget, "aligned budget fills its last page"
	# The terminal context fits ...
	_validate_page_capacity(_gpu_pages(seq), seq.current_context_length, "terminal")
	# ... and one token beyond it does not.
	with pytest.raises(AssertionError):
		_validate_page_capacity(
			_gpu_pages(seq), seq.current_context_length + 1, "terminal+1"
		)


def test_every_row_sees_the_synchronous_context_vector():
	"""Lookahead changes when the host reads, not what the device gets."""
	script = [[i, 100 + i] for i in range(1, 12)]
	_compare_modes(
		lambda: [FakeSeq("1", 192, 8), FakeSeq("2", 256, 11)],
		script,
		decision_interval=3,
	)


# --------------------------------------------------------------------------
# the loop's own arithmetic, so the driver above cannot drift from it
# --------------------------------------------------------------------------
def _decode_loop_source():
	worker = (
		Path(__file__).resolve().parents[1] / "batchgen" / "batchgen_worker.py"
	)
	tree = ast.parse(worker.read_text())
	fn = next(n for n in ast.walk(tree)
	          if isinstance(n, ast.FunctionDef) and n.name == "decoding_continuous")
	return ast.unparse(fn)


def test_the_loop_builds_cache_seqlens_through_effective_cache_seqlens():
	src = _decode_loop_source()
	assert "effective_cache_seqlens(_host_ctx, _lookahead_offsets)" in src


def test_the_loop_offsets_the_host_kv_write_position_too():
	# The host mirror must write at the SAME index as the device, or it
	# overwrites the previous token instead of appending this one.
	src = _decode_loop_source()
	assert "seq.current_context_length + _la_off - 1" in src


def test_the_decode_assert_checks_the_effective_context():
	src = _decode_loop_source()
	assert "seq.current_context_length + _la_off > max_tokens" in src


def test_no_loop_exit_can_skip_the_flush():
	"""Every break/continue must sit inside the flushed boundary block.

	A step whose tokens were never bookkept is a lost token per row, so the
	exits are structurally confined to the one block that flushes first (plus
	the unconditional flush after the loop).
	"""
	worker = (
		Path(__file__).resolve().parents[1] / "batchgen" / "batchgen_worker.py"
	)
	tree = ast.parse(worker.read_text())
	fn = next(n for n in ast.walk(tree)
	          if isinstance(n, ast.FunctionDef) and n.name == "decoding_continuous")
	loop = next(n for n in fn.body
	            if isinstance(n, ast.While) and ast.unparse(n.test) == "decode_uuids")
	boundary = next(
		s for s in loop.body
		if isinstance(s, ast.If) and "force_host_boundary" in ast.unparse(s.test)
	)
	inside = {id(n) for n in ast.walk(boundary)}

	found = []

	def visit(node):
		if isinstance(node, (ast.Break, ast.Continue)):
			found.append(node)
			return
		if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
			for child in node.orelse:
				visit(child)
			return
		if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
		                     ast.Lambda, ast.ClassDef)):
			return
		for child in ast.iter_child_nodes(node):
			visit(child)

	for stmt in loop.body:
		visit(stmt)
	assert found, "expected the decode loop to have exits at all"
	outside = [n.lineno for n in found if id(n) not in inside]
	assert outside == [], (
		"decode-loop exit outside the flushed boundary block at line(s) "
		f"{outside}: a pending step's tokens would be dropped"
	)


def test_only_the_resident_sampling_path_defers_its_readback():
	"""Eager and graph-compare forwards keep the synchronous order."""
	worker = (
		Path(__file__).resolve().parents[1] / "batchgen" / "batchgen_worker.py"
	)
	tree = ast.parse(worker.read_text())
	fn = next(n for n in ast.walk(tree)
	          if isinstance(n, ast.FunctionDef) and n.name == "decoding_continuous")
	assigns = [
		n for n in ast.walk(fn)
		if isinstance(n, ast.Assign)
		and any(getattr(t, "id", None) == "_lookahead_eligible" for t in n.targets)
	]
	values = sorted(ast.unparse(n.value) for n in assigns)
	# One per-step reset to True, and one opt-out per synchronous forward
	# branch (graph-vs-eager compare, per-layer/eager).
	assert values == ["False", "False", "True"], values

	# Each opt-out sits next to its own _select_tokens call, and neither of
	# those calls uses the resident sampling tensors.
	blocks = [
		block
		for parent in ast.walk(fn)
		for field in ("body", "orelse", "finalbody")
		for block in [getattr(parent, field, None)]
		if isinstance(block, list)
	]
	for node in assigns:
		if ast.unparse(node.value) != "False":
			continue
		block = next(b for b in blocks if any(s is node for s in b))
		before = ast.unparse(block[block.index(node) - 1])
		assert "_select_tokens" in before, before
		assert "residency=" not in before, before
