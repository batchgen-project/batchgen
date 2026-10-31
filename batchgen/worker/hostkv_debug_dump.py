"""Diagnostic dump of a held session's Host KV pages, and its byte compare.

Never enabled in production. ``BATCHGEN_DEBUG_HOSTKV_DUMP_DIR`` turns it on;
``BATCHGEN_DEBUG_HOSTKV_DUMP_SESSION_PREFIX`` is then mandatory, so only the
sessions a probe names are dumped. The worker calls ``dump_held_pages`` on the
Host-KV owner rank when it retains a finished turn's pages for a session hold
(``BatchGenWorker._retain_session_pages_for_hold``), after the turn's decode
appends were retired, so the bytes are the ones a later resume would load.

One dump is a directory ``<dir>/<session_id>/hold<NNN>_<request_id>/``:

* ``meta.json`` -- session, request, rank, ``kv_tokens``, ``page_tokens`` and,
  per Host region (prefix-cache group: 0 = MLA latent, 1 = DSA indexer keys),
  its page ids in token order, layer count, bytes of one layer's page and
  dtype;
* ``g<group>.bin`` -- page-major raw bytes: for each held page in token
  order, for each layer, the ``layer_page_bytes`` the Host view's
  ``k_page_ptr(layer, page)`` points at. That is the address and length the
  view's own H2D (``async_load_prefix_pages_to_device``) copies from.

``compare_dumps`` / ``python -m batchgen.worker.hostkv_debug_dump compare A B``
compares two dumps region by region, page by page over a common token range
(whole pages, then the leading rows of a partial page), and names the region,
page, layer and first differing byte of every unequal page.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

ENV_DIR = "BATCHGEN_DEBUG_HOSTKV_DUMP_DIR"
ENV_PREFIX = "BATCHGEN_DEBUG_HOSTKV_DUMP_SESSION_PREFIX"
META = "meta.json"
FORMAT_VERSION = 1


def config_from_env(environ: Mapping[str, str] = os.environ) -> Tuple[str, str]:
    """``(dir, session_prefix)``; ``("", "")`` when the dump is off.

    A directory without a prefix is refused: the dump must never fire for
    CUDA-graph capture, prime turns or unrelated sessions.
    """
    directory = environ.get(ENV_DIR, "")
    prefix = environ.get(ENV_PREFIX, "")
    if not directory:
        return "", ""
    if not prefix:
        raise ValueError(f"{ENV_DIR} requires {ENV_PREFIX}")
    return directory, prefix


@dataclass(frozen=True)
class RegionSpec:
    """One Host region of a held turn: what to read, and how much per page."""

    group: int
    view: Any
    num_layers: int
    layer_page_bytes: int
    dtype: str
    page_ids: Tuple[int, ...]


def region_spec(*, group: int, view, geometry, page_tokens: int, page_ids) -> RegionSpec:
    """A region's dump spec from its GPU KV geometry; refuses what it cannot read.

    Only K-only regions (MLA latent, DSA indexer keys) with the identity layer
    map and a contiguous page layout are dumped; anything else is refused by
    name rather than dumped with the wrong length.
    """
    import torch

    if int(getattr(geometry, "num_v_heads", 0) or 0) != 0:
        raise RuntimeError(
            f"Host KV dump: group {group} has a V cache "
            f"({geometry.num_v_heads} V heads); only K-only regions are dumped"
        )
    layer_map = getattr(geometry, "logical_to_physical_layer", None)
    num_layers = int(geometry.num_layers)
    if layer_map is not None and [int(x) for x in layer_map] != list(range(num_layers)):
        raise RuntimeError(
            f"Host KV dump: group {group} maps engine layers to other physical "
            "Host layers; only the identity layer map is dumped"
        )
    if bool(getattr(view, "is_page_split", False)):
        raise RuntimeError(
            f"Host KV dump: group {group} is a page-split region; its scales "
            "live outside the K page and would not be dumped"
        )
    element = torch.empty((), dtype=geometry.kv_dtype).element_size()
    layer_page_bytes = (
        int(page_tokens) * int(geometry.num_k_heads) * int(geometry.k_head_dim) * element
    )
    if layer_page_bytes <= 0 or num_layers <= 0:
        raise RuntimeError(
            f"Host KV dump: group {group} has an empty page "
            f"(layers={num_layers}, layer_page_bytes={layer_page_bytes})"
        )
    return RegionSpec(
        group=int(group), view=view, num_layers=num_layers,
        layer_page_bytes=layer_page_bytes, dtype=str(geometry.kv_dtype),
        page_ids=tuple(int(p) for p in page_ids),
    )


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(name))


def _page_stride(view) -> int:
    """Address distance between consecutive page ids of one layer (layout check)."""
    return int(view.k_page_ptr(0, 1)) - int(view.k_page_ptr(0, 0))


def dump_held_pages(
    root: str,
    *,
    session_id: str,
    request_id: str,
    rank: int,
    kv_tokens: int,
    page_tokens: int,
    regions: Sequence[RegionSpec],
) -> str:
    """Write one held turn's Host pages of every region; returns the dump dir.

    Written into a ``.partial`` directory and renamed when complete, so a
    reader never sees half a dump.
    """
    if not regions:
        raise RuntimeError("Host KV dump: no Host region named")
    needed = -(-int(kv_tokens) // int(page_tokens))
    for spec in regions:
        if len(spec.page_ids) != needed:
            raise RuntimeError(
                f"Host KV dump: group {spec.group} names {len(spec.page_ids)} "
                f"page(s) for {kv_tokens} tokens ({needed} pages of {page_tokens})"
            )
    session_dir = os.path.join(root, _safe(session_id))
    os.makedirs(session_dir, exist_ok=True)
    index = sum(1 for name in os.listdir(session_dir) if name.startswith("hold"))
    final = os.path.join(session_dir, f"hold{index:03d}_{_safe(request_id)}")
    partial = os.path.join(session_dir, f".hold{index:03d}_{_safe(request_id)}.partial")
    os.makedirs(partial)
    meta_regions = []
    for spec in regions:
        stride = _page_stride(spec.view)
        if stride < spec.layer_page_bytes:
            raise RuntimeError(
                f"Host KV dump: group {spec.group} pages are {stride} B apart "
                f"but one layer's page is {spec.layer_page_bytes} B"
            )
        name = f"g{spec.group}.bin"
        n = spec.layer_page_bytes
        with open(os.path.join(partial, name), "wb") as out:
            for page in spec.page_ids:
                for layer in range(spec.num_layers):
                    ptr = int(spec.view.k_page_ptr(layer, page))
                    if not ptr:
                        raise RuntimeError(
                            f"Host KV dump: group {spec.group} layer {layer} "
                            f"page {page} has no Host address"
                        )
                    out.write((ctypes.c_char * n).from_address(ptr))
        meta_regions.append({
            "group": spec.group, "file": name, "num_layers": spec.num_layers,
            "layer_page_bytes": spec.layer_page_bytes, "page_stride_bytes": stride,
            "dtype": spec.dtype, "page_ids": list(spec.page_ids),
        })
    meta = {
        "format_version": FORMAT_VERSION, "session_id": str(session_id),
        "request_id": str(request_id), "rank": int(rank),
        "kv_tokens": int(kv_tokens), "page_tokens": int(page_tokens),
        "layout": "page-major: for page in page_ids: for layer: layer_page_bytes",
        "regions": meta_regions,
    }
    with open(os.path.join(partial, META), "w") as out:
        json.dump(meta, out, indent=1)
    os.rename(partial, final)
    return final


def load_meta(path: str) -> Dict[str, Any]:
    with open(os.path.join(path, META)) as stream:
        meta = json.load(stream)
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path}: unknown Host KV dump format {meta.get('format_version')}")
    return meta


def _bf16_max_abs(a, b) -> Optional[float]:
    import numpy as np

    if len(a) % 2:
        return None
    ua = np.frombuffer(a, dtype=np.uint16).astype(np.uint32) << 16
    ub = np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16
    fa, fb = ua.view(np.float32), ub.view(np.float32)
    with np.errstate(invalid="ignore"):
        return float(np.nanmax(np.abs(fa - fb)))


def compare_dumps(a_dir: str, b_dir: str, *, tokens: Optional[int] = None) -> Dict[str, Any]:
    """Byte compare of two dumps over their first ``tokens`` tokens.

    ``tokens`` defaults to the smaller ``kv_tokens``; pass the common token
    prefix explicitly when the two sessions' outputs may differ. Whole pages
    are compared in full (every layer); a trailing partial page compares only
    its first ``tokens % page_tokens`` rows of each layer.
    """
    import numpy as np

    ma, mb = load_meta(a_dir), load_meta(b_dir)
    page_tokens = int(ma["page_tokens"])
    if int(mb["page_tokens"]) != page_tokens:
        raise ValueError(f"page_tokens differ: {page_tokens} vs {mb['page_tokens']}")
    limit = min(int(ma["kv_tokens"]), int(mb["kv_tokens"]))
    common = limit if tokens is None else int(tokens)
    if common <= 0 or common > limit:
        raise ValueError(
            f"compare tokens {common} outside (0, {limit}] "
            f"(kv_tokens {ma['kv_tokens']} / {mb['kv_tokens']})"
        )
    whole, tail_rows = divmod(common, page_tokens)
    ra = {int(r["group"]): r for r in ma["regions"]}
    rb = {int(r["group"]): r for r in mb["regions"]}
    if sorted(ra) != sorted(rb):
        raise ValueError(f"region groups differ: {sorted(ra)} vs {sorted(rb)}")
    result: Dict[str, Any] = {
        "a": a_dir, "b": b_dir, "tokens": common, "page_tokens": page_tokens,
        "whole_pages": whole, "tail_rows": tail_rows, "regions": {},
    }
    for group in sorted(ra):
        ga, gb = ra[group], rb[group]
        for key in ("num_layers", "layer_page_bytes", "dtype"):
            if ga[key] != gb[key]:
                raise ValueError(f"group {group} {key} differs: {ga[key]} vs {gb[key]}")
        layers, lpb = int(ga["num_layers"]), int(ga["layer_page_bytes"])
        if lpb % page_tokens:
            raise ValueError(f"group {group}: {lpb} B per page is not whole rows")
        row = lpb // page_tokens
        page_bytes = layers * lpb
        fa = np.memmap(os.path.join(a_dir, ga["file"]), dtype=np.uint8, mode="r")
        fb = np.memmap(os.path.join(b_dir, gb["file"]), dtype=np.uint8, mode="r")
        for f, g in ((fa, ga), (fb, gb)):
            if f.size != len(g["page_ids"]) * page_bytes:
                raise ValueError(
                    f"group {group}: {f.size} B on disk for {len(g['page_ids'])} pages "
                    f"of {page_bytes} B"
                )
        pages = whole + (1 if tail_rows else 0)
        equal, same_ids, diffs = 0, 0, []
        for index in range(pages):
            rows = page_tokens if index < whole else tail_rows
            pa = fa[index * page_bytes:(index + 1) * page_bytes].reshape(layers, lpb)[:, :rows * row]
            pb = fb[index * page_bytes:(index + 1) * page_bytes].reshape(layers, lpb)[:, :rows * row]
            same_ids += int(ga["page_ids"][index] == gb["page_ids"][index])
            if np.array_equal(pa, pb):
                equal += 1
                continue
            mask = pa != pb
            layer = int(np.flatnonzero(mask.any(axis=1))[0])
            offset = int(np.flatnonzero(mask[layer])[0])
            entry = {
                "page_index": index, "page_id_a": int(ga["page_ids"][index]),
                "page_id_b": int(gb["page_ids"][index]), "rows": rows,
                "first_diff_layer": layer, "first_diff_offset_in_layer_page": offset,
                "first_diff_token": index * page_tokens + offset // row,
                "first_diff_offset_in_page": layer * lpb + offset,
                "layers_differing": int(mask.any(axis=1).sum()),
                "bytes_differing": int(mask.sum()),
            }
            if "bfloat16" in str(ga["dtype"]):
                entry["max_abs_diff_bf16"] = _bf16_max_abs(
                    np.ascontiguousarray(pa).tobytes(), np.ascontiguousarray(pb).tobytes()
                )
            diffs.append(entry)
        result["regions"][str(group)] = {
            "dtype": ga["dtype"], "num_layers": layers, "layer_page_bytes": lpb,
            "pages_compared": pages, "pages_equal": equal, "pages_differing": len(diffs),
            "same_page_ids": same_ids, "diffs": diffs,
        }
    result["all_equal"] = all(
        r["pages_differing"] == 0 for r in result["regions"].values()
    )
    return result


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    cmp_parser = sub.add_parser("compare", help="byte compare two Host KV dumps")
    cmp_parser.add_argument("a")
    cmp_parser.add_argument("b")
    cmp_parser.add_argument("--tokens", type=int, default=None)
    cmp_parser.add_argument("--max-diffs", type=int, default=20,
                            help="diff entries printed per region (all are counted)")
    args = parser.parse_args(argv)
    result = compare_dumps(args.a, args.b, tokens=args.tokens)
    for region in result["regions"].values():
        region["diffs_shown"] = min(len(region["diffs"]), args.max_diffs)
        region["diffs"] = region["diffs"][:args.max_diffs]
    print(json.dumps(result, indent=1))
    return 0 if result["all_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
