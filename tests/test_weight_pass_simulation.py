"""Deterministic simulation of weight streaming on one node, CPU-only.

Workers share one node daemon. Each worker streams L layers x E routed
experts through a GPU ring of R slots. The daemon serves the workers whose
modules the node pulls (a local worker reads the node's own store) from a
staging ring of D slots, ordered per layer in superchunks of S across those
workers exactly as ``BuildSchedule`` in
``core/Weights_Storage/distributed_weight_daemon.cpp`` orders them. Like
``PrefetchLoop`` it posts chunks from a bounded window over the cyclic
schedule into any free one of its D / S chunk-slots, a chunk-slot freeing
once its worker released all of its modules. A chunk an acquire waits for
is demanded and posted first; otherwise the oldest chunk whose owner is
under its in-flight cap goes. Each chunk enters the window once per cycle,
in schedule order, so every module is fetched exactly once per pass.

The pass-armed model mirrors ``core/GPU_Weight_Buffer/weight_slot_table.h``:
the producer of a worker produces a position only while its pass is armed,
a consumer declares a random idle subset per layer and leases the rest,
arms the next pass early (the explicit carry) at random, cancels a carried
pass or a pass whose forward raised, and one worker per trial delays its
begin and end by up to a full pass. Every trial must finish, and every pass
end must see the conditions ``end_weight_pass`` asserts.

The pipelined model is that producer as ``HtoD_Worker`` in
``core/HtoD_Engine/HtoD_Engine.cu`` runs it: up to A copies in flight on one
stream, landed oldest first, each daemon lease released before its slot
turns PRESENT. Its daemon socket carries one request at a time: an acquire
that is not ready waits in bounded slices and is resent, and the worker
sends no release until it returns. Fetching a remote module only with
nothing in flight finishes every trial in schedule order; a blocking
acquire issued while a lease is held deadlocks the node. The try-acquire
producer finishes every trial in schedule order too: it also fetches a
remote module with copies in flight, but only with a no-wait request the
daemon answers at once (``kAcquireNoWait``), and tries one that was not
ready again only after the next landing.

The free-running model is the engine before passes: a cyclic producer that
reads ahead across forwards, stopped at each worker's own boundary and
reinstalled from the head of its list at the next forward, as the worker's
phase switch does. Under the same skew it strands the daemon FIFO.

No core import: this models the contract, the compiled table is tested in
``tests/test_weight_slot_table.py``.
"""

import random
from collections import Counter, defaultdict, deque

import pytest


N = 8  # workers on the node
L = 4  # layers
E = 8  # streamed experts per layer and worker
S = 2  # daemon superchunk
R = 4  # GPU ring slots per worker
D = 16  # daemon ring slots, fewer than one worker's modules per pass
P = 5  # passes per trial
A = 4  # copies in flight per producer: HtoD_Engine kMaxCopiesInFlight
ACQUIRE_SLICE = 3  # rounds a daemon acquire waits before it answers retry

FULL_PASS_ROUNDS = 4 * L * E
MAX_ROUNDS = 100_000
TRIALS = 24

IDLE_RATE = 0.3
CARRY_RATE = 0.5
CANCEL_RATE = 0.3
RAISE_RATE = 0.1


def module_names():
    return [
        f"routed_expert_{layer}_{expert}"
        for layer in range(L)
        for expert in range(E)
    ]


def daemon_schedule(workers=N, superchunk=S, local=()):
    """One pass of the daemon FIFO: per layer, superchunks across workers.

    A local worker's modules are read from the node's own store and never
    enter the daemon.
    """
    schedule = []
    for layer in range(L):
        for offset in range(0, E, superchunk):
            for worker in range(workers):
                if worker in local:
                    continue
                for index in range(superchunk):
                    schedule.append(
                        (worker, f"routed_expert_{layer}_{offset + index}")
                    )
    return schedule


class Deadlock(Exception):
    pass


class Daemon:
    """The staging chunk-slots over the cyclic schedule.

    Global chunk g is entries [g * S, (g + 1) * S) of the cyclic schedule.
    A window holds the next unposted chunks in schedule order, at most
    2 * (outstanding + D / S) and never more than one pass of them. While
    fewer than ``outstanding`` chunks are in flight and a chunk-slot is free,
    the first demanded chunk of the window is posted, else the oldest one
    whose owner has fewer than 3 x its rails chunks in flight. A chunk-slot
    frees once every module of its chunk was released. An acquire that finds
    its module not ready demands its chunk; one past the window is demanded
    as the refill reaches it. A fetch takes ``fetch_polls`` calls of
    ``fetch``; 0 completes it at once. Two workers' shards share an owner,
    as with four nodes.
    """

    def __init__(self, slots=D, superchunk=S, workers=N, local=(), *,
                 outstanding=None, fetch_polls=0, rails=None):
        self.schedule = daemon_schedule(workers, superchunk, local)
        assert slots % superchunk == 0 and 0 < slots <= len(self.schedule)
        self.slots = slots
        self.superchunk = superchunk
        self.positions = slots // superchunk
        self.chunks_per_pass = len(self.schedule) // superchunk
        self.outstanding = (
            self.positions if outstanding is None else outstanding
        )
        self.capacity = min(
            2 * (self.outstanding + self.positions), self.chunks_per_pass
        )
        self.fetch_polls = fetch_polls
        self.rails = rails or {}  # owner -> rails; 1 if absent
        self.free = list(range(self.positions - 1, -1, -1))  # LIFO
        self.live = [0] * self.positions  # unreleased modules per chunk-slot
        self.chunk_slot = {}  # live global chunk -> its chunk-slot
        self.window = deque()  # [global chunk, demanded], schedule order
        self.next_chunk = 0  # the next global chunk the refill takes
        self.deferred = {}  # worker -> module its acquire waits for
        self.in_flight = {}  # global chunk -> {"polls", "demanded"}
        self.pool = {}  # entry index -> "fetching" | "ready" | "leased"
        self.posts = []  # {"chunk", "front", "demanded"} in post order
        self.peak_speculative = Counter()  # owner -> undemanded in flight
        self.stats = Counter()

    def entry(self, index):
        """(pass, worker, module) of a global FIFO index."""
        worker, name = self.schedule[index % len(self.schedule)]
        return index // len(self.schedule) + 1, worker, name

    def chunk_entries(self, chunk):
        return range(chunk * self.superchunk, (chunk + 1) * self.superchunk)

    def chunk_worker(self, chunk):
        return self.entry(chunk * self.superchunk)[1]

    def owner(self, chunk):
        return self.chunk_worker(chunk) // 2

    def cap(self, owner):
        return 3 * self.rails.get(owner, 1)

    def _holds(self, chunk, worker, name):
        return any(
            self.entry(index)[1:] == (worker, name)
            for index in self.chunk_entries(chunk)
        )

    def _owner_in_flight(self, owner, demanded=None):
        return sum(
            1 for chunk, flight in self.in_flight.items()
            if self.owner(chunk) == owner
            and (demanded is None or flight["demanded"] == demanded)
        )

    def _refill(self):
        while len(self.window) < self.capacity:
            chunk = self.next_chunk
            # Strict schedule order: each chunk enters once per cycle.
            assert not self.window or self.window[-1][0] < chunk
            self.next_chunk += 1
            demanded = False
            for worker, name in list(self.deferred.items()):
                if self._holds(chunk, worker, name):
                    del self.deferred[worker]
                    demanded = True
            if demanded:
                self.stats["demand_hits"] += 1
            self.window.append([chunk, demanded])

    def _select(self):
        for position, (_, demanded) in enumerate(self.window):
            if demanded:
                return position
        for position, (chunk, _) in enumerate(self.window):
            owner = self.owner(chunk)
            if self._owner_in_flight(owner) < self.cap(owner):
                return position
            self.stats["capped_skips"] += 1
        return None

    def _post(self, position):
        front = self.window[0][0]
        chunk, demanded = self.window[position]
        del self.window[position]
        slot = self.free.pop()
        assert self.live[slot] == 0
        for index in self.chunk_entries(chunk):
            # No module is fetched twice within a cycle.
            assert not any(
                self.entry(held)[1:] == self.entry(index)[1:]
                for held in self.pool
            ), (chunk, self.describe())
        state = "fetching" if self.fetch_polls else "ready"
        for index in self.chunk_entries(chunk):
            self.pool[index] = state
        if self.fetch_polls:
            self.in_flight[chunk] = {
                "polls": self.fetch_polls, "demanded": demanded,
            }
            owner = self.owner(chunk)
            self.peak_speculative[owner] = max(
                self.peak_speculative[owner],
                self._owner_in_flight(owner, demanded=False),
            )
        self.live[slot] = self.superchunk
        self.chunk_slot[chunk] = slot
        self.stats["expedited_posts"] += demanded
        self.posts.append(
            {"chunk": chunk, "front": front, "demanded": demanded}
        )

    def fetch(self):
        """Advance the fetches in flight, then post as ``PrefetchLoop`` does."""
        progressed = False
        for chunk in list(self.in_flight):
            progressed = True
            self.in_flight[chunk]["polls"] -= 1
            if self.in_flight[chunk]["polls"] == 0:
                del self.in_flight[chunk]
                for index in self.chunk_entries(chunk):
                    self.pool[index] = "ready"
        while len(self.in_flight) < self.outstanding:
            self._refill()
            if not self.free:
                self.stats["freelist_empty_polls"] += 1
                break
            position = self._select()
            if position is None:
                break  # every owner at its cap, nothing demanded
            self._post(position)
            progressed = True
        return progressed

    def _demand(self, worker, name):
        """``DemandChunkLocked``: posted chunks first, then the window."""
        if any(self._holds(chunk, worker, name) for chunk in self.chunk_slot):
            return  # posted: in flight or ready
        for pending in self.window:
            if self._holds(pending[0], worker, name):
                if not pending[1]:
                    pending[1] = True
                    self.stats["demand_hits"] += 1
                return
        if self.deferred.get(worker) != name:
            self.deferred[worker] = name
            self.stats["demand_misses"] += 1

    def acquire(self, worker, name):
        """Lease the oldest unleased entry of (worker, name) once ready.

        Not ready: its chunk is demanded, before any wait, as ``Acquire``
        does.
        """
        for index in sorted(self.pool):
            _, owner, module = self.entry(index)
            if owner == worker and module == name and self.pool[index] != "leased":
                if self.pool[index] == "ready":
                    self.pool[index] = "leased"
                    self.deferred.pop(worker, None)
                    return index
                break  # posted, still in flight
        self._demand(worker, name)
        return None

    def release(self, index):
        assert self.pool.pop(index) == "leased"
        chunk = index // self.superchunk
        slot = self.chunk_slot[chunk]
        self.live[slot] -= 1
        if self.live[slot] == 0:
            del self.chunk_slot[chunk]
            self.free.append(slot)

    def describe(self):
        return sorted(
            (self.entry(index), state) for index, state in self.pool.items()
        )


class Connection:
    """One worker's daemon socket: one request at a time, as ``HandleWorker``
    serves it.

    An acquire of a module that is not ready holds the socket for a bounded
    slice and answers retry; ``Weights_Storage`` resends at once under its
    daemon mutex. Until the acquire returns, the worker sends nothing else,
    so no lease it holds can be released.
    """

    def __init__(self, daemon, rank, slice_rounds=ACQUIRE_SLICE):
        self.daemon = daemon
        self.rank = rank
        self.slice_rounds = slice_rounds
        self.waiting = None  # module of the outstanding acquire
        self.waited = 0  # rounds into the current slice
        self.blocked = 0  # acquires that had to wait
        self.retries = 0

    def acquire(self, name):
        """The leased entry once ready; None while the acquire still waits."""
        assert self.waiting in (None, name)
        entry = self.daemon.acquire(self.rank, name)
        if entry is not None:
            self.waiting = None
            self.waited = 0
            return entry
        if self.waiting is None:
            self.waiting = name
            self.blocked += 1
        self.waited += 1
        if self.waited == self.slice_rounds:
            # The slice expired: retry is answered and the request resent.
            self.waited = 0
            self.retries += 1
        return None

    def try_acquire(self, name):
        """One ``kAcquireNoWait`` request: the leased entry, or None at once.

        The daemon answers without a slice, so the socket is free again and
        the worker may release a lease right after.
        """
        assert self.waiting is None
        return self.daemon.acquire(self.rank, name)

    def release(self, entry):
        assert self.waiting is None, (
            f"worker {self.rank} sent a release behind its acquire of "
            f"{self.waiting}"
        )
        self.daemon.release(entry)


def _sleep(rounds):
    for _ in range(rounds):
        yield "sleep"


class _Worker:
    def consume(self):
        if self.done:
            return "done"
        try:
            return next(self._consumer)
        except StopIteration:
            self.done = True
            return "acted"


class PassArmedWorker(_Worker):
    """One worker under pass arming: its slot table, producer and consumer."""

    def __init__(self, rank, daemon, rng, delay, *, idle_rate=IDLE_RATE,
                 carry_rate=CARRY_RATE, cancel_rate=CANCEL_RATE,
                 raise_rate=RAISE_RATE, hold_rounds=0):
        self.rank = rank
        self.daemon = daemon
        self.rng = rng
        self.delay = delay
        self.idle_rate = idle_rate
        self.carry_rate = carry_rate
        self.cancel_rate = cancel_rate
        self.raise_rate = raise_rate
        self.hold_rounds = hold_rounds  # rounds the consumer holds a lease
        self.names = module_names()
        self.position = {name: i for i, name in enumerate(self.names)}
        # None, or {"name", "pass", "state", "rol"} (rol: release on landing)
        self.slots = [None] * R
        self.cursor = [1, 0]
        self.open = deque()
        self.next_pass = 1
        self.pending = set()  # (pass, name)
        self.copied = Counter()
        self.skipped = Counter()
        self.acquired = Counter()  # daemon acquisitions per pass
        self.in_flight = None  # (slot index, daemon entry)
        self.summaries = []
        self.cancelled = []
        self.done = False
        self._consumer = self._consume()

    # -- slot table ------------------------------------------------------- #

    def _slot(self, name, pass_id):
        for index, slot in enumerate(self.slots):
            if slot is not None and slot["name"] == name and slot["pass"] == pass_id:
                return index, slot
        return None, None

    def _produced(self, pass_id, position):
        cursor_pass, cursor_position = self.cursor
        return cursor_pass > pass_id or (
            cursor_pass == pass_id and cursor_position > position
        )

    def begin(self):
        assert len(self.open) < 2, "a third open pass"
        pass_id = self.next_pass
        self.next_pass += 1
        self.open.append(pass_id)
        return pass_id

    def skip(self, names):
        pass_id = self.open[0]
        for name in names:
            index, slot = self._slot(name, pass_id)
            if slot is not None:
                assert slot["state"] in ("PRESENT", "FILLING") and not slot["rol"]
                if slot["state"] == "PRESENT":
                    self.slots[index] = None
                else:
                    slot["rol"] = True
                continue
            assert (pass_id, name) not in self.pending
            assert not self._produced(pass_id, self.position[name]), (
                f"{name} already consumed in pass {pass_id}"
            )
            self.pending.add((pass_id, name))

    def lease_ready(self, name):
        pass_id = self.open[0]
        _, slot = self._slot(name, pass_id)
        if slot is None:
            # Never a wait for a module that can no longer arrive.
            assert (pass_id, name) not in self.pending
            assert not self._produced(pass_id, self.position[name])
            return False
        assert slot["state"] in ("FILLING", "PRESENT") and not slot["rol"]
        return slot["state"] == "PRESENT"

    def lease(self, name):
        _, slot = self._slot(name, self.open[0])
        assert slot["state"] == "PRESENT"
        slot["state"] = "LEASED"

    def release(self, name):
        index, slot = self._slot(name, self.open[0])
        assert slot["state"] == "LEASED"
        self.slots[index] = None

    def cancel(self, pass_id):
        assert pass_id in self.open
        assert not any(
            slot is not None and slot["pass"] == pass_id and slot["state"] == "LEASED"
            for slot in self.slots
        )
        for index, slot in enumerate(self.slots):
            if slot is None or slot["pass"] != pass_id:
                continue
            if slot["state"] == "PRESENT":
                self.slots[index] = None
            elif slot["state"] == "FILLING":
                slot["rol"] = True
        for position, name in enumerate(self.names):
            if not self._produced(pass_id, position):
                self.pending.add((pass_id, name))
        self.cancelled.append(pass_id)

    # -- producer --------------------------------------------------------- #

    def produce(self):
        if self.in_flight is not None:
            index, entry = self.in_flight
            self.in_flight = None
            # The daemon lease is released before the copy lands.
            self.daemon.release(entry)
            slot = self.slots[index]
            self.copied[slot["pass"]] += 1
            if slot["rol"]:
                self.slots[index] = None
            else:
                slot["state"] = "PRESENT"
            return "acted"
        pass_id, position = self.cursor
        if pass_id not in self.open:
            return "blocked"  # parked until the pass is armed
        name = self.names[position]
        skip = (pass_id, name) in self.pending
        free = None
        if not skip:
            free = next(
                (i for i, slot in enumerate(self.slots) if slot is None), None
            )
            if free is None:
                return "blocked"
        entry = self.daemon.acquire(self.rank, name)
        if entry is None:
            return "blocked"
        # The producer and the daemon FIFO stay on the same pass.
        assert self.daemon.entry(entry)[0] == pass_id
        self.acquired[pass_id] += 1
        if skip:
            self.daemon.release(entry)
            self.pending.discard((pass_id, name))
            self.skipped[pass_id] += 1
        else:
            self.slots[free] = {
                "name": name, "pass": pass_id, "state": "FILLING", "rol": False,
            }
            self.in_flight = (free, entry)
        self.cursor[1] += 1
        if self.cursor[1] == len(self.names):
            self.cursor = [pass_id + 1, 0]
        return "acted"

    # -- consumer --------------------------------------------------------- #

    def _end(self, pass_id):
        assert self.open and self.open[0] == pass_id
        while self.cursor[0] <= pass_id or any(
            slot is not None and slot["pass"] == pass_id and slot["state"] == "FILLING"
            for slot in self.slots
        ):
            yield "blocked"
        held = [
            slot for slot in self.slots
            if slot is not None and slot["pass"] == pass_id
        ]
        assert held == [], f"worker {self.rank} pass {pass_id} holds {held}"
        assert not any(pending == pass_id for pending, _ in self.pending)
        # The daemon ring drains at the boundary except the carried prefix.
        for index, state in self.daemon.pool.items():
            entry_pass, owner, _ = self.daemon.entry(index)
            if owner != self.rank:
                continue
            assert entry_pass > pass_id, (self.rank, pass_id, entry_pass, state)
            if state == "leased":
                assert entry_pass == pass_id + 1 and pass_id + 1 in self.open
        assert self.copied[pass_id] + self.skipped[pass_id] == L * E
        carried = sum(
            1 for slot in self.slots
            if slot is not None and slot["pass"] == pass_id + 1
        )
        assert carried <= R
        if pass_id + 1 not in self.open:
            assert carried == 0 and self.acquired[pass_id + 1] == 0
        self.open.popleft()
        self.summaries.append({
            "pass": pass_id,
            "carried": carried,
            "copied": self.copied[pass_id],
            "skipped": self.skipped[pass_id],
        })
        yield "acted"

    def _consume(self):
        pass_id = 1
        while pass_id <= P:
            if pass_id not in self.open:
                yield from _sleep(self.delay("begin", pass_id))
                assert self.begin() == pass_id
                yield "acted"
            raise_at = (
                self.rng.randrange(L) if self.rng.random() < self.raise_rate else None
            )
            carry = (
                pass_id < P and raise_at is None
                and self.rng.random() < self.carry_rate
            )
            for layer in range(L):
                if layer == raise_at:
                    break
                if carry and layer == L - 1:
                    assert self.begin() == pass_id + 1
                    yield "acted"
                names = self.names[layer * E:(layer + 1) * E]
                idle = [name for name in names if self.rng.random() < self.idle_rate]
                if idle:
                    self.skip(idle)
                    yield "acted"
                for name in names:
                    if name in idle:
                        continue
                    while not self.lease_ready(name):
                        yield "blocked"
                    self.lease(name)
                    yield "acted"
                    yield from _sleep(self.hold_rounds)
                    self.release(name)
                    yield "acted"
            if raise_at is not None:
                # The forward raised: everything it did not lease is skipped.
                self.cancel(pass_id)
                yield "acted"
            yield from _sleep(self.delay("end", pass_id))
            cancel_next = carry and self.rng.random() < self.cancel_rate
            early = cancel_next and self.rng.random() < 0.5
            if early:
                self.cancel(pass_id + 1)
                yield "acted"
            yield from self._end(pass_id)
            if cancel_next:
                if not early:
                    self.cancel(pass_id + 1)
                    yield "acted"
                yield from self._end(pass_id + 1)
                pass_id += 2
            else:
                pass_id += 1


class PipelinedWorker(PassArmedWorker):
    """The pass-armed worker with up to A copies in flight.

    One copy stream: copies complete in submit order, the oldest unfinished
    one advancing a round per round, copy_rounds rounds each, whether or not
    the producer thread is free. Each produce lands the completed copies,
    oldest first, releasing each daemon lease before its slot turns PRESENT,
    then submits while fewer than max_in_flight are in flight. A local
    module is read at once. A remote module takes a blocking acquire, during
    which the producer lands and releases nothing; lease_safe issues one only
    with nothing in flight, so the acquire never waits holding a lease.
    try_acquire (with lease_safe) also fetches a remote module while copies
    are in flight, but only with a no-wait request: a module not ready yet
    stays at the cursor, and is tried again only after the next landing.
    """

    def __init__(self, rank, daemon, rng, delay, *, local, copy_rounds,
                 lease_safe=True, try_acquire=False, max_in_flight=A,
                 **kwargs):
        super().__init__(rank, daemon, rng, delay, **kwargs)
        assert lease_safe or not try_acquire
        self.local = local
        self.copy_rounds = copy_rounds
        self.lease_safe = lease_safe
        self.try_acquire = try_acquire
        self.max_in_flight = max_in_flight
        self.connection = Connection(daemon, rank)
        self.in_flight = deque()  # {"slot", "entry", "rounds"}, oldest first
        self.fetching = None  # (pass, name, reserved slot) being acquired
        self.fetched = defaultdict(list)  # pass -> modules in fetch order
        self.peak_in_flight = 0
        self.tried = False  # a try was not ready; none again until a landing
        self.tries = Counter()  # "ready" / "not_ready" no-wait requests

    def _run_copy_stream(self):
        for copy in self.in_flight:
            if copy["rounds"]:
                copy["rounds"] -= 1
                return True
        return False

    def _land(self):
        landed = False
        while self.in_flight and self.in_flight[0]["rounds"] == 0:
            copy = self.in_flight.popleft()
            if copy["entry"] is not None:
                # Released before the landing: once no slot of a pass is
                # FILLING, no lease of it is held either.
                self.connection.release(copy["entry"])
            slot = self.slots[copy["slot"]]
            self.copied[slot["pass"]] += 1
            if slot["rol"]:
                self.slots[copy["slot"]] = None
            else:
                slot["state"] = "PRESENT"
            landed = True
            self.tried = False
        return landed

    def _claim(self, entry):
        pass_id, name, slot = self.fetching
        self.fetching = None
        if entry is not None:
            # The producer and the daemon FIFO stay on the same pass.
            assert self.daemon.entry(entry)[0] == pass_id
        self.acquired[pass_id] += 1
        self.fetched[pass_id].append(name)
        if (pass_id, name) in self.pending:
            # A skip, possibly declared while the acquire waited: the lease
            # is released at once and a reserved slot is abandoned.
            if entry is not None:
                self.connection.release(entry)
            self.pending.discard((pass_id, name))
            self.skipped[pass_id] += 1
        else:
            self.slots[slot] = {
                "name": name, "pass": pass_id, "state": "FILLING", "rol": False,
            }
            self.in_flight.append(
                {"slot": slot, "entry": entry, "rounds": self.copy_rounds}
            )
            self.peak_in_flight = max(self.peak_in_flight, len(self.in_flight))
        self.cursor[1] += 1
        if self.cursor[1] == len(self.names):
            self.cursor = [pass_id + 1, 0]

    def produce(self):
        acted = self._run_copy_stream()
        if self.fetching is not None:
            # Still inside the blocking acquire.
            entry = self.connection.acquire(self.fetching[1])
            if entry is None:
                return "acted" if acted else "blocked"
            self._claim(entry)
            acted = True
        if self._land():
            acted = True
        while len(self.in_flight) < self.max_in_flight:
            pass_id, position = self.cursor
            if pass_id not in self.open:
                break  # parked until the pass is armed
            try_only = (
                not self.local and bool(self.in_flight) and self.lease_safe
            )
            if try_only and (not self.try_acquire or self.tried):
                break  # wait for an empty pipeline, or for the next landing
            name = self.names[position]
            slot = None
            if (pass_id, name) not in self.pending:
                slot = next(
                    (i for i, held in enumerate(self.slots) if held is None),
                    None,
                )
                if slot is None:
                    break
            entry = None
            if try_only:
                entry = self.connection.try_acquire(name)
                if entry is None:
                    # Not ready: the slot goes back, the cursor stays, and
                    # nothing counts as progress.
                    self.tries["not_ready"] += 1
                    self.tried = True
                    break
                self.tries["ready"] += 1
                self.fetching = (pass_id, name, slot)
            else:
                # Never a blocking acquire while a lease is held.
                assert not (self.lease_safe and not self.local and self.in_flight)
                self.fetching = (pass_id, name, slot)
                if not self.local:
                    entry = self.connection.acquire(name)
                    if entry is None:
                        acted = True
                        break  # the producer thread now waits in the acquire
            acted = True
            self._claim(entry)
        return "acted" if acted else "blocked"


class FreeRunningWorker(_Worker):
    """The engine before passes: a cyclic producer with a per-worker stop."""

    def __init__(self, rank, daemon, end_delay):
        self.rank = rank
        self.daemon = daemon
        self.end_delay = end_delay
        self.names = module_names()
        self.slots = [None] * R  # PRESENT module names
        self.position = 0
        self.running = True
        self.dropped = 0  # read-ahead modules the reinstall discarded
        self.done = False
        self._consumer = self._consume()

    def produce(self):
        if not self.running:
            return "blocked"
        free = next((i for i, slot in enumerate(self.slots) if slot is None), None)
        if free is None:
            return "blocked"
        name = self.names[self.position]
        entry = self.daemon.acquire(self.rank, name)
        if entry is None:
            return "blocked"
        self.daemon.release(entry)
        self.slots[free] = name
        self.position = (self.position + 1) % len(self.names)
        return "acted"

    def _consume(self):
        for forward in range(P):
            if forward:
                # The next forward reinstalls the queue from its head and
                # resets the ring, dropping what the producer read ahead.
                self.dropped += sum(slot is not None for slot in self.slots)
                self.slots = [None] * R
                self.position = 0
                self.running = True
                yield "acted"
            for name in self.names:
                while name not in self.slots:
                    yield "blocked"
                self.slots[self.slots.index(name)] = None
                yield "acted"
            yield from _sleep(self.end_delay(forward))
            # stop_h2d_worker at this worker's own boundary.
            self.running = False
            yield "acted"


def run(workers, daemon, order_rng):
    """Rounds in which every actor may act once, in a seeded random order."""
    for rounds in range(MAX_ROUNDS):
        if all(worker.done for worker in workers):
            return rounds
        progressed = daemon.fetch()
        sleeping = False
        order = list(workers)
        order_rng.shuffle(order)
        for worker in order:
            steps = [worker.produce, worker.consume]
            order_rng.shuffle(steps)
            for step in steps:
                outcome = step()
                progressed |= outcome == "acted"
                sleeping |= outcome == "sleep"
                progressed |= daemon.fetch()
        if not progressed and not sleeping:
            raise Deadlock(
                f"round {rounds}: done={[worker.done for worker in workers]}, "
                f"daemon={daemon.describe()}"
            )
    raise AssertionError(f"no completion within {MAX_ROUNDS} rounds")


def _skew(delay_rng, skewed):
    def delay(kind, pass_id):
        return delay_rng.randint(0, FULL_PASS_ROUNDS) if skewed else 0
    return delay


def run_pass_armed_trial(seed):
    order_rng = random.Random(seed)
    skewed_rank = order_rng.randrange(N)
    daemon = Daemon()
    workers = [
        PassArmedWorker(
            rank,
            daemon,
            random.Random(seed * 1009 + rank),
            _skew(random.Random(seed * 7919 + rank), rank == skewed_rank),
        )
        for rank in range(N)
    ]
    run(workers, daemon, order_rng)
    return workers


@pytest.mark.parametrize("seed", range(TRIALS))
def test_pass_armed_streaming_finishes_and_drains_every_boundary(seed):
    # Every end-of-pass condition is asserted inside the model as it runs.
    workers = run_pass_armed_trial(seed)
    for worker in workers:
        assert [summary["pass"] for summary in worker.summaries] == list(
            range(1, P + 1)
        )
        for summary in worker.summaries:
            assert summary["copied"] + summary["skipped"] == L * E
            assert summary["carried"] <= R
        assert worker.pending == set()
        assert not worker.open
        assert worker.in_flight is None
        assert all(slot is None for slot in worker.slots)


def test_trials_exercise_carry_cancel_skips_and_skew():
    totals = Counter()
    for seed in range(TRIALS):
        for worker in run_pass_armed_trial(seed):
            totals["carried"] += sum(s["carried"] for s in worker.summaries)
            totals["skipped"] += sum(s["skipped"] for s in worker.summaries)
            totals["cancelled"] += len(worker.cancelled)
    assert totals["carried"] > 0
    assert totals["skipped"] > 0
    assert totals["cancelled"] > 0


def _late_end(rank):
    # Worker 0 ends every forward a full pass after it finished consuming.
    return lambda *_: FULL_PASS_ROUNDS if rank == 0 else 0


@pytest.mark.parametrize("seed", range(4))
def test_free_running_producer_with_per_worker_stop_deadlocks_under_skew(seed):
    daemon = Daemon()
    workers = [
        FreeRunningWorker(rank, daemon, _late_end(rank)) for rank in range(N)
    ]
    with pytest.raises(Deadlock):
        run(workers, daemon, random.Random(seed))
    # While its end was late, worker 0 read the next forward's head ahead.
    # The reinstall dropped those copies, but the daemon FIFO had already
    # served them, so the worker waits for modules that come only a pass
    # later, behind its own unacquired entries that fill the daemon ring.
    assert workers[0].dropped > 0


@pytest.mark.parametrize("seed", range(4))
def test_pass_armed_streaming_survives_the_same_skew(seed):
    daemon = Daemon()
    workers = [
        PassArmedWorker(
            rank,
            daemon,
            random.Random(seed * 1009 + rank),
            (lambda rank: lambda kind, pass_id: (
                FULL_PASS_ROUNDS if rank == 0 and kind == "end" else 0
            ))(rank),
            idle_rate=0.0,
            carry_rate=0.0,
            cancel_rate=0.0,
            raise_rate=0.0,
        )
        for rank in range(N)
    ]
    run(workers, daemon, random.Random(seed))
    for worker in workers:
        assert [summary["pass"] for summary in worker.summaries] == list(
            range(1, P + 1)
        )
        assert all(summary["carried"] == 0 for summary in worker.summaries)
        assert all(summary["copied"] == L * E for summary in worker.summaries)


# (workers, local workers, superchunk, chunk-slots): D = chunk-slots x S.
PIPELINE_CONFIGS = [
    (8, 2, 2, 4),  # six remote workers over four chunk-slots
    (8, 0, 2, 8),
    (6, 0, 1, 3),
    (4, 1, 2, 2),
    (4, 2, 1, 6),  # more chunk-slots than remote workers
    (3, 0, 2, 1),  # one chunk-slot: every chunk waits for the one before
]
PIPELINE_SEEDS = range(6)
NO_RANDOM_EVENTS = {
    "idle_rate": 0.0, "carry_rate": 0.0, "cancel_rate": 0.0, "raise_rate": 0.0,
}
# Ranks 0 and 1 local; ranks 2-7 own chunks 0-5 of every superchunk round,
# so each remote rank's next chunk is six chunks after its current one.
DEADLOCK_CONFIG = (8, 2, 2, 4)


def build_pipelined_trial(config, seed, *, lease_safe=True, try_acquire=False,
                          skew=True, **rates):
    workers_count, local_count, superchunk, positions = config
    local = range(local_count)
    daemon = Daemon(positions * superchunk, superchunk, workers_count, local)
    order_rng = random.Random(seed)
    skewed_rank = order_rng.randrange(workers_count) if skew else None
    workers = []
    for rank in range(workers_count):
        speed_rng = random.Random(seed * 104729 + rank)
        workers.append(PipelinedWorker(
            rank,
            daemon,
            random.Random(seed * 1009 + rank),
            _skew(random.Random(seed * 7919 + rank), rank == skewed_rank),
            local=rank in local,
            copy_rounds=speed_rng.randint(1, 4),
            hold_rounds=speed_rng.randint(0, 2),
            lease_safe=lease_safe,
            try_acquire=try_acquire,
            **rates,
        ))
    return workers, daemon, order_rng


def run_pipelined_trial(config, seed, **kwargs):
    workers, daemon, order_rng = build_pipelined_trial(config, seed, **kwargs)
    run(workers, daemon, order_rng)
    return workers, daemon


def _assert_schedule_order(workers, daemon):
    for worker in workers:
        order = [
            name for owner, name in daemon.schedule if owner == worker.rank
        ]
        assert order == (module_names() if not worker.local else [])
        assert sorted(worker.fetched) == list(range(1, P + 1))
        for pass_id in range(1, P + 1):
            assert worker.fetched[pass_id] == module_names()
        assert [summary["pass"] for summary in worker.summaries] == list(
            range(1, P + 1)
        )
        for summary in worker.summaries:
            assert summary["copied"] + summary["skipped"] == L * E
            assert summary["carried"] <= R
        assert worker.pending == set()
        assert not worker.open
        assert not worker.in_flight and worker.fetching is None
        assert all(slot is None for slot in worker.slots)
    assert "leased" not in daemon.pool.values()


@pytest.mark.parametrize("seed", PIPELINE_SEEDS)
@pytest.mark.parametrize("config", PIPELINE_CONFIGS)
def test_pipelined_producer_finishes_in_schedule_order(config, seed):
    # Every end-of-pass condition is asserted inside the model as it runs.
    workers, daemon = run_pipelined_trial(config, seed)
    _assert_schedule_order(workers, daemon)
    for worker in workers:
        if not worker.local:
            # A remote module is only fetched into an empty pipeline.
            assert worker.peak_in_flight <= 1
            assert not worker.tries


@pytest.mark.parametrize("seed", PIPELINE_SEEDS)
@pytest.mark.parametrize("config", PIPELINE_CONFIGS)
def test_try_acquire_producer_finishes_in_schedule_order(config, seed):
    # The model asserts that no blocking acquire is issued with a copy, and
    # so possibly a lease, in flight.
    workers, daemon = run_pipelined_trial(config, seed, try_acquire=True)
    _assert_schedule_order(workers, daemon)
    for worker in workers:
        assert worker.peak_in_flight <= A
        if worker.local:
            assert not worker.tries


def test_pipelined_trials_exercise_pipelining_waits_and_carry():
    totals = Counter()
    for config in PIPELINE_CONFIGS:
        workers, _ = run_pipelined_trial(config, 0)
        for worker in workers:
            if worker.local:
                totals["local_peak"] = max(
                    totals["local_peak"], worker.peak_in_flight
                )
            totals["blocked"] += worker.connection.blocked
            totals["retries"] += worker.connection.retries
            totals["carried"] += sum(s["carried"] for s in worker.summaries)
            totals["skipped"] += sum(s["skipped"] for s in worker.summaries)
            totals["cancelled"] += len(worker.cancelled)
    assert totals["local_peak"] > 1
    assert totals["blocked"] > 0
    assert totals["retries"] > 0
    assert totals["carried"] > 0
    assert totals["skipped"] > 0
    assert totals["cancelled"] > 0


@pytest.mark.parametrize("seed", range(4))
def test_blocking_acquire_while_holding_a_lease_deadlocks(seed):
    workers, daemon, order_rng = build_pipelined_trial(
        DEADLOCK_CONFIG, seed, lease_safe=False, skew=False,
        **NO_RANDOM_EVENTS,
    )
    with pytest.raises(Deadlock):
        run(workers, daemon, order_rng)
    # The daemon posted chunks 0-3 first, those of ranks 2-5, into its four
    # chunk-slots. Each rank leased both modules of its chunk, then, both
    # still in flight, waited in the acquire of its next chunk. That chunk is
    # demanded, but a chunk-slot frees only once its chunk is released, which
    # no rank can send from inside the acquire. Ranks 6 and 7 wait for their
    # demanded first chunk with no chunk-slot free either.
    holding = {
        worker.rank for worker in workers
        if worker.fetching is not None
        and any(copy["entry"] is not None for copy in worker.in_flight)
    }
    assert holding == {2, 3, 4, 5}
    for rank in holding:
        # The copies completed: only their landing, and its release, is owed.
        assert all(copy["rounds"] == 0 for copy in workers[rank].in_flight)
    for rank in (6, 7):
        assert workers[rank].fetching is not None
        assert not workers[rank].in_flight
    assert workers[0].done and workers[1].done
    assert not daemon.free
    assert {daemon.chunk_worker(chunk) for chunk in daemon.chunk_slot} == holding


def test_lease_safe_producer_finishes_the_deadlock_configuration():
    workers, _ = run_pipelined_trial(
        DEADLOCK_CONFIG, 0, skew=False, **NO_RANDOM_EVENTS
    )
    assert all(worker.done for worker in workers)
    assert all(
        worker.peak_in_flight > 1 for worker in workers if worker.local
    )


@pytest.mark.parametrize("seed", range(4))
def test_try_acquire_producer_finishes_the_deadlock_configuration(seed):
    # The configuration where a blocking acquire behind a lease deadlocks:
    # the remote ranks now keep copies in flight too, and still finish.
    workers, daemon, order_rng = build_pipelined_trial(
        DEADLOCK_CONFIG, seed, try_acquire=True, skew=False,
        **NO_RANDOM_EVENTS,
    )
    run(workers, daemon, order_rng)
    _assert_schedule_order(workers, daemon)
    for worker in workers:
        assert worker.done
        assert worker.peak_in_flight > 1
        if not worker.local:
            # Each chunk's second module is ready with its first in flight;
            # the next chunk is not, until other ranks released theirs.
            assert worker.tries["ready"] > 0
            assert worker.tries["not_ready"] > 0


def test_try_acquire_trials_pipeline_remote_ranks_and_retry_per_landing():
    totals = Counter()
    for config in PIPELINE_CONFIGS:
        workers, daemon = run_pipelined_trial(config, 0, try_acquire=True)
        totals["expedited"] += daemon.stats["expedited_posts"]
        for worker in workers:
            if worker.local:
                continue
            totals["remote_peak"] = max(
                totals["remote_peak"], worker.peak_in_flight
            )
            totals["ready"] += worker.tries["ready"]
            totals["not_ready"] += worker.tries["not_ready"]
            totals["blocked"] += worker.connection.blocked
            totals["cancelled"] += len(worker.cancelled)
    assert totals["remote_peak"] > 1
    assert totals["ready"] > 0
    assert totals["not_ready"] > 0
    # With nothing in flight the fetch still waits for the daemon.
    assert totals["blocked"] > 0
    assert totals["cancelled"] > 0
    # A rank waiting for a chunk gets it posted ahead of the window's order.
    assert totals["expedited"] > 0


class StreamingConsumer:
    """A worker that acquires its modules in order straight from the daemon.

    It holds at most ``leases`` leases and releases the oldest once it was
    held ``hold_rounds`` steps; None never releases. Each step asks for the
    next module before releasing, as a producer with copies in flight does.
    """

    def __init__(self, daemon, rank, *, hold_rounds, leases=1, passes=1,
                 modules=None):
        self.daemon = daemon
        self.rank = rank
        self.hold_rounds = hold_rounds
        self.leases = leases
        names = module_names() if modules is None else modules
        self.per_pass = len(names)
        self.order = names * passes
        self.position = 0
        self.held = deque()  # [entry, steps left], oldest first

    @property
    def done(self):
        return self.position == len(self.order) and not self.held

    def step(self):
        acted = False
        if self.position < len(self.order) and len(self.held) < self.leases:
            entry = self.daemon.acquire(self.rank, self.order[self.position])
            if entry is not None:
                # Every module once per pass, in the daemon's pass.
                assert self.daemon.entry(entry)[0] == (
                    self.position // self.per_pass + 1
                )
                self.held.append([entry, self.hold_rounds])
                self.position += 1
                acted = True
        if self.held and self.held[0][1] is not None:
            if self.held[0][1] == 0:
                self.daemon.release(self.held.popleft()[0])
                return "acted"
            self.held[0][1] -= 1
            return "acted" if acted else "sleep"
        return "acted" if acted else "blocked"


def drive(daemon, consumers):
    """Rounds in which each consumer steps once, in rank order."""
    for rounds in range(MAX_ROUNDS):
        if all(consumer.done for consumer in consumers):
            return rounds
        progressed = daemon.fetch()
        sleeping = False
        for consumer in consumers:
            outcome = consumer.step()
            progressed |= outcome == "acted"
            sleeping |= outcome == "sleep"
            progressed |= daemon.fetch()
        if not progressed and not sleeping:
            raise Deadlock(f"round {rounds}: daemon={daemon.describe()}")
    raise AssertionError(f"no completion within {MAX_ROUNDS} rounds")


def _assert_fetched_once_per_cycle(daemon, passes):
    posted = Counter(post["chunk"] for post in daemon.posts)
    assert all(count == 1 for count in posted.values())
    assert set(range(passes * daemon.chunks_per_pass)) <= set(posted)


def test_waiting_fast_worker_is_posted_ahead_of_older_speculative_chunks():
    # Worker 0 holds each module 6 steps; worker 1 asks for its next module
    # while still holding one. Two chunk-slots: chunks 0 and 1 are posted
    # first. Worker 1 finds chunk 3 not ready and demands it; when worker 1
    # releases chunk 1, chunk 3 is posted ahead of worker 0's older chunk 2.
    passes = 2
    daemon = Daemon(slots=4, superchunk=2, workers=2)
    slow = StreamingConsumer(daemon, 0, hold_rounds=6, passes=passes)
    fast = StreamingConsumer(
        daemon, 1, hold_rounds=1, leases=2, passes=passes
    )
    drive(daemon, [slow, fast])

    expedited = [post for post in daemon.posts if post["demanded"]]
    first = expedited[0]
    assert (first["chunk"], first["front"]) == (3, 2)
    assert daemon.chunk_worker(first["chunk"]) == 1
    assert daemon.chunk_worker(first["front"]) == 0
    # Worker 1 keeps jumping worker 0's speculative chunks.
    assert sum(
        post["chunk"] > post["front"] and daemon.chunk_worker(post["chunk"]) == 1
        for post in expedited
    ) > 1
    assert daemon.stats["demand_hits"] > 0
    _assert_fetched_once_per_cycle(daemon, passes)


def test_a_never_released_chunk_pins_only_its_own_chunk_slot():
    # Worker 0 leases its first chunk and never releases it, nor asks for
    # more. A positional ring would stop workers 1 and 2 after their first
    # chunk, once the tail wrapped onto that chunk. The free list keeps
    # serving them until worker 0's chunks, the held one and the speculative
    # ones it never acquires, take every chunk-slot.
    daemon = Daemon(slots=6, superchunk=2, workers=3)
    stuck = StreamingConsumer(
        daemon, 0, hold_rounds=None, leases=2,
        modules=module_names()[:2],
    )
    others = [
        StreamingConsumer(daemon, rank, hold_rounds=0, passes=2)
        for rank in (1, 2)
    ]
    with pytest.raises(Deadlock):
        drive(daemon, [stuck, *others])
    assert stuck.position == 2 and len(stuck.held) == 2
    assert not daemon.free
    assert len(daemon.chunk_slot) == daemon.positions
    assert {daemon.chunk_worker(chunk) for chunk in daemon.chunk_slot} == {0}
    for consumer in others:
        assert not consumer.done
        # Served far past what the staging ring holds at once.
        assert consumer.position > daemon.slots


def test_speculative_chunks_in_flight_stay_under_the_owner_cap():
    # Owners 0 (workers 0, 1) and 1 (workers 2, 3) have 1 and 2 rails, so
    # at most 3 and 6 undemanded chunks in flight; a demanded chunk is never
    # held back by the cap.
    rails = {0: 1, 1: 2}
    daemon = Daemon(
        slots=24, superchunk=2, workers=4, outstanding=12, fetch_polls=6,
        rails=rails,
    )
    consumers = [
        StreamingConsumer(
            daemon, rank, hold_rounds=3 if rank == 0 else 1, leases=2,
        )
        for rank in range(4)
    ]
    drive(daemon, consumers)
    for owner in rails:
        assert daemon.peak_speculative[owner] <= daemon.cap(owner)
    # The cap bound: owner 0 reached it and a candidate was passed over.
    assert daemon.peak_speculative[0] == daemon.cap(0)
    assert daemon.stats["capped_skips"] > 0
    _assert_fetched_once_per_cycle(daemon, 1)
