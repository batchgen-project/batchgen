# Prefix Cache Eviction Implementation Plan

## 目标

在 prefix reuse 端到端输出验证通过后，补齐 Host Prefix Cache 的 eviction 能力。目标是：当历史 prefix cache 页面逐渐占满 host KV 空间时，新的 prefill allocation 可以自动回收冷 prefix cache 页面，而不是直接分配失败或只能依赖 `ClearPrefixCache()` 全量清空。

当前已验证的是 deterministic GPT-OSS 路径下 full reuse / partial reuse / miss 的输出 token 级一致性；这不等价于 logits 或 KV tensor 级 bitwise 一致。eviction 设计不能依赖“首 token 一样”作为 KV 正确性证明，后续验证需要增加 logits/KV debug compare。

非目标：

- 不改变 active sequence 的 host KV 生命周期语义。
- 不做 token-level eviction，仍然保持 page-level chained hash prefix index。
- 不把 prefix cache eviction 和现有 host KV active sequence eviction 混成同一套策略。二者可以协作，但职责不同。
- 不改变 decode batch selection。prefix reuse 只影响 prefill/host allocation；进入 decode 前 GPU KV 已经是完整逻辑上下文，decode 不应该因为 KV 来源于 prefix cache 而拆小 batch 或隔离 request。

## 当前实现状态

当前 prefix cache 的核心数据结构大致是：

```text
HostPrefixCache
  PrefixPageKey(namespace, page_size, page_index, parent_page_hash, page_token_hash)
      -> PrefixPageEntry(host_page_id, page_chain_hash, pin_count)

HostPagedKVBackend
  page_sequence_refs[page]  // active logical sequence references
  page_prefix_pins[page]    // prefix cache ownership pins

HostKVPageTable
  sequence_id -> shared_prefix_pages + private_pages
```

当前生命周期：

```text
prefill commit
  -> HostPrefixCache::CommitPages()
  -> backend.PinPrefixPage(page)

new request lookup
  -> HostPrefixCache::Lookup()
  -> allocate private suffix pages
  -> backend.AttachSequencePages(shared_prefix_pages)
  -> HostKVPageTable.RegisterOrUpdate(shared_prefix_pages, private_pages)

sequence completion
  -> backend.ReleaseSequenceLogical(...)
  -> detach sequence refs
  -> prefix pins remain

manual/global cleanup only
  -> HostPrefixCache::Clear()
  -> backend.UnpinPrefixPage(page)
```

当前缺口：完成的 batch 只释放 sequence refs，不释放 prefix pins；因此历史 prefix cache 页面会长期占住 host pages。随着 batch 增多，prefix cache 会越来越大，最终影响新的 private suffix page allocation。

当前实现还需要注意几个具体事实：

- `HostPrefixCache::Lookup()` 只做 chained-hash lookup 并统计 hit/miss，还没有 access epoch / LRU 元数据。
- `HostPrefixCache::CommitPages()` 只为新插入的完整页调用 `PinPrefixPage()`；已有 entry 不会重复 pin，因此当前 `PrefixPageEntry::pin_count` 实际上是一条 entry 的 ownership pin。
- `HostPagedKVWorkerView::AllocatePagesForSequencesWithPrefix()` 当前逐 request 执行 lookup、private page allocation、shared page attach、page-table register。引入 eviction/retry 前必须改成 batch-level plan-then-commit，避免部分 request 成功后失败造成 ref 泄漏。
- `ClearPrefixCache()` 是当前唯一会批量删除 prefix entries 并 `UnpinPrefixPage()` 的路径。

## 顶层设计

### 1. 两级 Eviction 职责

```text
Prefix Cache Eviction
  对象：历史 prefix cache entry
  动作：从 HostPrefixCache index 删除 entry，并 UnpinPrefixPage(page)
  结果：如果 page_sequence_refs == 0 且 page_prefix_pins == 0，该 page 回到 backend free pool

Host KV Sequence Eviction
  对象：active / on-hold sequence
  动作：释放 sequence pages，sequence 进入 EVICTED，后续 recompute
  结果：为 active serving 让空间
```

prefix cache eviction 只负责清历史 cache，不应该直接改变 active sequence 状态。即使某个 prefix page 被 active sequence 引用，evict prefix entry 也只是减少 `page_prefix_pins`；页面仍由 `page_sequence_refs` 保活，不影响当前 sequence 的 host/GPU page table。

### 2. Eviction 触发点

第一版采用 allocation-time eviction：

```text
AllocatePagesForSequencesWithPrefix(requests)
  1. lookup all requests
  2. calculate required private suffix pages
  3. if free_pages < required_private_pages:
       evict cold prefix cache pages until free_pages reaches target
  4. allocate private pages transactionally
  5. attach protected shared prefix pages
  6. register combined page table
```

prefix cache 默认可以占满所有未被 active sequence 引用的 host pages。不设置 prefix cache 自身容量上限，也不做 proactive budget eviction；只有新的 allocation 需要空间时，才 pressure-driven 地回收冷 prefix entries。

第一版优先保证 allocate 不失败，先不引入后台线程或后台清理。

### 3. Eviction 粒度：Leaf-First Page Eviction

prefix index 是 chained hash：

```text
page0(ROOT) -> page1(hash(page0)) -> page2(hash(page1)) -> ...
```

如果直接删除中间页，后续子页无法再被 lookup 命中，但仍可能占用 prefix pin，形成不可达泄漏。因此 eviction 应采用 leaf-first：

```text
root page
  └── page 1
      └── page 2
          └── page 3  <- first eviction candidate
```

删除 leaf 后，父节点可能变成新的 leaf。这样可以优先丢掉最长、最冷、最具体的后缀页面，同时保留更通用的短 prefix。

### 4. Eviction 策略

第一版策略：LRU leaf eviction。

每个 `PrefixPageEntry` 增加：

```cpp
uint64_t insert_epoch;
uint64_t last_access_epoch;
uint64_t hit_count;
uint32_t child_count;
```

更新规则：

- `CommitPages()` 插入 entry 时设置 `insert_epoch = last_access_epoch = ++epoch`。
- `CommitPages()` 命中已有 entry 时不重复 `PinPrefixPage()`；可以刷新 `last_access_epoch` / `hit_count`，但必须保持 one-entry-one-prefix-pin 语义。
- `Lookup()` 每命中一个 entry，更新 `last_access_epoch = ++epoch`，`hit_count++`。
- leaf candidate 必须满足 `child_count == 0`。
- victim 排序按 `last_access_epoch ASC`，相同则 `insert_epoch ASC`。

后续可选策略：

- LRU + hit_count 权重，保护高频短 prefix。
- namespace-level quota，避免单模型/单 workload 占满所有 prefix pages。
- min-prefix-pages-to-keep，避免 eviction 后 reuse 完全退化。

### 5. Protected Pages

allocation-time eviction 不能把当前 request batch 已经 lookup 命中的 shared prefix page 淘汰掉，否则本 batch 会从 hit 变成 miss，甚至产生 attach stale page 风险。

因此 eviction API 需要支持 protected page set：

```cpp
struct PrefixEvictionOptions {
    size_t target_free_pages;
    size_t max_entries_to_scan;
    std::unordered_set<int32_t> protected_pages;
};
```

eviction 跳过：

- 当前 allocation lookup 命中的 pages。
- 未来可扩展为跳过 hot pages / pinned-by-policy pages。

如果 protected pages 导致无法释放足够空间，第一版行为应明确失败并返回可诊断错误；第二版可做 per-request fallback，把低收益 hit 降级为 miss 后重试。

### 6. Backend Refcount 语义

eviction 的核心安全条件：

```text
Remove prefix entry
  -> backend.UnpinPrefixPage(page)
     if page_sequence_refs == 0 && page_prefix_pins == 0:
        page becomes free
     else:
        page remains allocated until sequence refs release
```

需要新增 backend 查询能力，至少用于 stats/debug：

```cpp
struct HostPageRefState {
    int32_t page;
    uint32_t sequence_refs;
    uint32_t prefix_pins;
    bool free_if_unpinned_once;
};
```

第一版可以不依赖该查询做正确性，只在每轮 unpin 后重新读取 aggregate `num_free_pages`，直到达到 target。但测试和日志需要能解释为什么 evicted entries 没有立刻释放页面。

### 7. Rank Cache 失效

Python 侧有 `_prefix_reuse_prompt_rank_cache`，用于把相同 prompt 路由到已有 prefix 的 rank。eviction 后该缓存可能指向已经没有 prefix entry 的 rank。

需要增加 prefix cache generation：

```text
HostPrefixCache.eviction_epoch++
GetPrefixCacheStats().eviction_epoch
```

Python 侧策略：

```python
if stats.eviction_epoch != self._prefix_reuse_rank_cache_epoch:
    self._prefix_reuse_prompt_rank_cache.clear()
    self._prefix_reuse_rank_cache_epoch = stats.eviction_epoch
```

第一版也可以更保守：只要 `--enable-prefix-reuse` 打开并发生任意 eviction，就清空整个 prompt rank cache。

rank cache 失效主要是命中率/性能问题，不是正确性问题：如果缓存指向的 rank 已经没有对应 prefix entry，本次 allocation 会自然变成 miss 并走 full/private prefill；但它可能错过其它 rank 上仍存在的 prefix，因此需要清空以恢复 rank-affinity 命中率。

### 8. Decode 调度透明性

prefix cache eviction 不应该参与 decode batch selection，也不应该因为某条 sequence 曾经使用过 reused prefix 而限制 decode batch size。

正确边界是：

```text
prefill/allocation:
  cached prefix pages + private suffix pages -> complete logical host/GPU KV

decode:
  read complete page_table + cache_seqlens
  do not branch on prefix_shared_tokens for scheduling
```

eviction 删除的是历史 prefix index entry 和 prefix pin。active sequence 的 page table、`prefix_shared_tokens` 记录、GPU KV materialization 语义都不能被同步修改；否则会把 cache 管理策略泄漏到 decode 计算路径，重新引入 batch-shape drift 风险。

## 设计图

### Allocation-Time Eviction

```text
new prefill batch
      |
      v
lookup prefix cache for all requests
      |
      v
compute:
  protected_shared_pages
  total_private_pages_required
      |
      v
free pages enough?
      |
      +-- yes --> allocate private pages -> attach shared pages
      |
      +-- no --> evict cold leaf entries excluding protected pages
                   |
                   v
                free pages enough?
                   |
                   +-- yes --> allocate private pages -> attach shared pages
                   |
                   +-- no --> controlled allocation failure / fallback policy
```

### Refcount Safety

```text
Prefix cache entry removed
      |
      v
UnpinPrefixPage(page)
      |
      +-- sequence_refs == 0
      |      page returned to free pool
      |
      +-- sequence_refs > 0
             active sequence still owns logical page
             page returns to free pool after sequence release
```

### Leaf-First Eviction

```text
Before:
  A0
   └─ A1
       └─ A2
  B0
   └─ B1

Leaf candidates:
  A2, B1

After evict A2:
  A0
   └─ A1   <- now leaf candidate
  B0
   └─ B1
```

## API Plan

### C++: HostPrefixCache

Add metadata:

```cpp
struct PrefixPageEntry {
    PrefixPageKey key;
    uint64_t page_chain_hash;
    int32_t host_page_id;
    int32_t page_size;
    uint64_t token_validation_hash;
    uint32_t pin_count;
    uint64_t insert_epoch;
    uint64_t last_access_epoch;
    uint64_t hit_count;
    uint32_t child_count;
};
```

Add eviction result:

```cpp
struct PrefixEvictionResult {
    size_t requested_free_pages;
    size_t entries_removed;
    size_t prefix_pins_released;
    size_t pages_immediately_freed;
    size_t protected_entries_skipped;
    size_t active_ref_entries_removed;
    bool reached_target;
};
```

Add methods:

```cpp
PrefixEvictionResult EvictLeafPages(
    const PrefixEvictionOptions& options,
    const UnpinCallback& on_unpin,
    const FreePageCountCallback& free_pages);

PrefixCacheStats Stats() const;  // include eviction counters + epoch
```

Implementation notes:

- Maintain `child_count` during insert/delete.
- Use stable parent key lookup or parent chain hash mapping to decrement parent `child_count`.
- Start with O(N) scan for cold leaves. Prefix cache eviction is not on the token hot path.
- Do not expose an entry to `Lookup()` after its prefix pin has been removed.
- Keep the ownership model explicit: one live prefix entry owns one prefix pin. Do not increment `pin_count` on repeated `CommitPages()` for an existing key unless the implementation also stores and decrements every additional owner.

### C++: HostPagedKVBackend

Add diagnostic page ref query:

```cpp
HostPageRefState PageRefState(int32_t page) const;
std::vector<HostPageRefState> PageRefStates(const std::vector<int32_t>& pages) const;
```

Optional helper:

```cpp
size_t FreePageCount() const;
```

### C++: HostPagedKVWorkerView

Add:

```cpp
PrefixEvictionResult EvictPrefixCacheUntilFree(
    size_t target_free_pages,
    const std::unordered_set<int32_t>& protected_pages);
```

Change `AllocatePagesForSequencesWithPrefix()`:

```text
1. Ensure sequences registered
2. Lookup all requests
3. Build protected_pages from all lookup hits
4. Compute total private_pages_required
5. Evict until enough free pages
6. Allocate all private pages transactionally
7. Attach shared pages
8. Register HostKVPageTable records
9. Roll back all attached/allocated pages on any failure
```

Important: make the batch allocation transactional. The current implementation processes requests one by one; with eviction/retry, partial success followed by failure would be hard to reason about.

### Python: BatchGenWorker

Add prefix cache eviction stats logging:

```text
[PREFIX_EVICT] target_free=... entries_removed=... pins_released=...
[PREFIX_EVICT] protected_skipped=... immediate_free=... reached_target=...
```

Add rank cache invalidation:

```text
if prefix cache eviction_epoch changes:
  clear _prefix_reuse_prompt_rank_cache
```

Eviction enablement:

```text
No new server flag. Prefix cache eviction is enabled automatically when
--enable-prefix-reuse is enabled, and remains unreachable when prefix reuse is
disabled.
```

Recommended first-version defaults:

- No reserve-pages or max-pages knobs. Prefix cache may fill free host pages and is evicted only under allocation pressure.

## Detailed TODO / Checklist

### Milestone 0: Preconditions

- [x] Prefix reuse exactness is green for target GPT-OSS path.
- [x] Clarify validation scope: output token-level exactness is required; logits/KV tensor compare is recommended before claiming bitwise cache equivalence.
- [x] Default `--enable-prefix-reuse` disabled behavior is still byte-for-byte identical to `origin/main`.
- [x] Current prefix cache stats are understood: entries, pages with prefix pins, prefix pin increments/decrements, host pages saved.
- [x] Decide whether eviction is guarded behind a new flag or automatically enabled under `--enable-prefix-reuse`.
- [x] Confirm decode scheduling remains prefix-transparent: no decode batch isolation or size change based on `prefix_shared_tokens`.

### Milestone 1: Prefix Cache Metadata

- [x] Add `insert_epoch`, `last_access_epoch`, `hit_count`, `child_count` to `PrefixPageEntry`.
- [x] Add global `access_epoch` and `eviction_epoch` to `HostPrefixCache`.
- [x] Update `Lookup()` to refresh access metadata for every matched page.
- [x] Update `CommitPages()` to initialize access metadata for inserted pages.
- [x] Update `CommitPages()` existing-entry path to refresh metadata without adding another prefix pin.
- [x] Maintain parent `child_count` on insert.
- [x] Extend `PrefixCacheStats` with eviction counters:
- [x] `eviction_epoch`
- [x] `eviction_runs`
- [x] `evicted_entries`
- [x] `evicted_prefix_pins`
- [x] `eviction_protected_skips`
- [x] `eviction_target_failures`

### Milestone 2: Leaf Eviction Primitive

- [x] Implement cold leaf candidate scan.
- [x] Skip protected pages.
- [x] Remove selected leaf entries and decrement parent `child_count`.
- [x] Call `backend.UnpinPrefixPage(page)` exactly once per removed cache pin.
- [x] Keep `prefix_pin_increments - prefix_pin_decrements == live prefix pins`.
- [x] Add deterministic tie-breaking for tests.
- [x] Implement `Clear()` via the same unpin accounting path or keep it consistent with eviction stats.

### Milestone 3: Backend Diagnostics

- [x] Add page-level ref state query in `HostPagedKVBackend`.
- [x] Expose aggregate free page count without requiring full stats formatting.
- [x] Add debug logging for pages evicted but not immediately freed because `sequence_refs > 0`.
- [x] Add assertions for prefix pin underflow and impossible free-page transitions.

### Milestone 4: Allocation Integration

- [x] Refactor `AllocatePagesForSequencesWithPrefix()` into plan-then-commit phases.
- [x] Lookup all requests before allocating any private pages.
- [x] Build `protected_pages` from all lookup hits.
- [x] Compute total private page requirement for the whole batch.
- [x] Evict cold prefix pages until `free_pages >= private_pages_required`.
- [x] Re-check free pages after eviction.
- [x] Allocate all private pages transactionally.
- [x] Attach shared pages only after eviction is complete.
- [x] Register page table records only after attach + private allocation succeeds.
- [x] Roll back private pages and attached shared pages on any exception.
- [x] Return eviction summary in allocation result or expose it via stats.
- [x] Preserve existing no-eviction behavior when enough free pages are available.

### Milestone 5: Rank Cache Invalidation

- [x] Expose `eviction_epoch` through Python stats.
- [x] Track `_prefix_reuse_rank_cache_epoch` in `BatchGenWorker`.
- [x] Clear `_prefix_reuse_prompt_rank_cache` on eviction epoch change.
- [x] Add log line when rank cache is cleared due to prefix eviction.
- [x] Test same prompt after eviction clears stale prompt-rank affinity before the next routing pass.
- [x] Verify stale rank cache is a miss/performance fallback only and cannot corrupt output.

### Milestone 6: Active Sequence Safety

- [x] Test evicting a prefix entry while an active sequence still references that page.
- [x] Verify active sequence can still decode/load host KV after prefix entry removal.
- [x] Verify page becomes free only after the active sequence releases sequence refs.
- [ ] Verify `ReleaseSequencePages()` with shared prefix pages remains idempotent and refcount-safe.
- [ ] Verify host KV sequence eviction and prefix cache eviction can happen in either order.
- [x] Verify prefix eviction does not mutate active sequence `prefix_shared_tokens`, per-sequence allocation metadata, or decode page-table rows.

### Milestone 6.5: Decode Transparency Regression

- [x] Ensure `_prefix_reuse_decode_rank_blocked()` or equivalent scheduling code does not isolate reused-prefix requests.
- [ ] Run mixed full/partial/miss decode with prefix reuse enabled and compare batch sizing/logs against no-reuse where practical.
- [ ] Add a regression test or log assertion that prefix eviction counters do not affect decode candidate selection.

### Milestone 7: Policy Controls

- [x] Do not add a new eviction enablement arg; eviction is automatic under `--enable-prefix-reuse`.
- [x] Do not add reserve-pages or max-pages flags; prefix cache is allowed to fill available host pages.
- [x] Keep eviction policy inside `HostPagedKVWorkerView`; no extra config propagation is required for the automatic policy.
- [x] Ensure default behavior remains unchanged when prefix reuse is disabled.
- [x] Document automatic eviction behavior in `docs/server-flags.md`.

### Milestone 8: Tests

- [x] Unit/helper: rank cache clears when `eviction_epoch` changes.
- [x] Integration: `HostPrefixCache` leaf-first LRU evicts only leaves.
- [x] Integration: evicting leaf preserves shorter prefix lookup.
- [x] Integration: protected pages are skipped.
- [x] Integration: eviction stats and pin counters are balanced.
- [x] Integration: fill prefix cache, release sequences, allocate new request under pressure, eviction frees pages and allocation succeeds.
- [x] Integration: active-ref page eviction removes cache entry but does not free page until sequence release.
- [ ] Integration: allocation rollback after forced failure restores page refs and prefix pins.
- [x] Integration: rank cache invalidates after eviction.
- [ ] E2E: warm prefixes, force small host KV budget, run mixed full/partial/miss batch with eviction enabled.
- [ ] E2E: compare no-prefix and prefix+eviction outputs for exactness on deterministic GPT-OSS test set.
- [ ] Debug: optional logits diff for selected partial/miss rows before and after eviction pressure.
- [ ] Debug: optional KV page diff for warm prefix load + suffix offload on a small deterministic batch.

### Milestone 9: Observability

- [x] Add log summary per eviction run.
- [x] Add prefix cache stats to existing worker stats dump.
- [x] Add counters for lookup hit/miss.
- [x] Add counters for attached shared pages.
- [x] Add counters for prefix pages inserted.
- [x] Add counters for prefix pages evicted.
- [x] Add counters for immediate pages freed.
- [x] Add counters for evicted pages still held by sequence refs.
- [x] Add counters for allocation failures after eviction.
- [ ] Add a small debug command or Python accessor to dump top cold/hot prefix entries.

### Milestone 10: Remote Validation

- [ ] Run unit/integration tests locally or in container.
- [ ] Run remote import audit after C++ binding/API changes.
- [ ] Run small GPT-OSS-120B smoke:
- [ ] warm 5-10 prefixes
- [ ] mixed 200 requests
- [ ] constrained host KV to force eviction
- [ ] Run larger GPT-OSS-120B validation:
- [ ] warm 50 prefixes
- [ ] mixed 1000 requests
- [ ] host KV budget small enough to trigger multiple eviction waves
- [ ] Verify no-prefix vs prefix+eviction output exactness.
- [ ] Verify repeated prefix+eviction runs are exact.
- [ ] Verify no leaked GPU or host KV processes after run cleanup.

## Failure Modes To Guard

- Evicting a parent page while children remain indexed, causing unreachable pinned pages.
- Removing prefix entry before protecting current allocation hits, causing hit-to-miss races.
- Unpinning prefix page twice, causing prefix pin underflow.
- Evicting prefix pages but not clearing Python prompt-rank cache, causing stale rank routing.
- Allocation failure after partially attaching shared pages, causing sequence ref leaks.
- Active sequence decode reading a page that was freed because sequence refs were not held.
- Prefix cache eviction hiding real host KV capacity pressure from active sequence eviction.
- Prefix eviction or prefix-hit metadata changing decode batch shape, causing BF16 batch-shape drift even when logical KV is correct.
- Treating output-token equality as proof that logits/KV tensors are identical.

## First Implementation Slice

Recommended first PR scope:

1. Implement leaf-first LRU eviction in `HostPrefixCache`.
2. Add pressure-driven eviction inside `AllocatePagesForSequencesWithPrefix()`.
3. Add stats and rank-cache invalidation.
4. Add unit/integration tests for refcount and allocation pressure.
5. Run small remote GPT-OSS validation with constrained host KV.

Defer namespace quota and hit-to-miss fallback until pressure-driven eviction is stable. Do not add proactive prefix-cache budgets unless a later workload proves they are necessary.
