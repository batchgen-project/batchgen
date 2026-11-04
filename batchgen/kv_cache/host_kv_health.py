"""One-line health summary of a host-KV region for the decode heartbeat.

The fields come from ``get_audited_stats()`` on a host-KV manager or worker
view: the page identity (used pages against the pages held by live sequence
entries plus prefix-resident pages), the sequence-table occupancy, and the
region's lifetime counters. A page leak shows as a delta that persists or grows
across lines; a table filling with tombstones shows as ``empty`` falling long
before any insert is refused.

``get_audited_stats()`` walks the whole region under both of its locks, so it
belongs on a periodic line only, never on a per-step path.
"""

from __future__ import annotations


def format_host_kv_health(region: str, stats) -> str:
	"""Format audited host-KV stats; plain ``get_stats()`` output is rejected."""
	audit = stats.audit
	if audit is None:
		raise ValueError(
			f"host-KV health line for region {region!r} needs get_audited_stats(); "
			"get_stats() carries no sequence-table audit"
		)
	delta = int(audit.page_identity_delta)
	identity = "ok" if delta == 0 else f"MISMATCH(delta={delta})"
	return (
		f"region={region} "
		f"pages: used={stats.num_used_pages}/{stats.num_total_pages} "
		f"live_seq={audit.live_sequence_pages} "
		f"resident={audit.prefix_resident_pages} identity={identity} | "
		f"table: live={audit.live_entries} tombstone={audit.tombstone_entries} "
		f"empty={audit.empty_entries} capacity={stats.sequence_table_capacity} | "
		f"lifetime: inserted={stats.sequences_inserted} "
		f"released={stats.sequences_released} "
		f"acquire_failures={stats.acquire_failures}"
	)
