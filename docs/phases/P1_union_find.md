# P1: Union-Find Decoder (core decoder #1)

**Time box:** 2:00  **Depends on:** P0  **Unlocks:** P3

## Goal
Implement the Delfosse–Nickerson Union-Find decoder on our `DecodingGraph`. Build an unweighted version first so it's simple and correct, then a weighted one, which matters for circuit noise.

## Concept primer (tell the human this before coding)
Think of each defect as a drop of ink on paper.
1. **Grow.** Every "unhappy" ink blot spreads outward by half an edge per step.
2. **Merge.** When two blots touch, they become one blot. That's the *union* in Union-Find. A data structure called *disjoint sets* makes "which blot am I in?" almost instant.
3. **Stop.** A blot is **happy** when it contains an *even* number of defects, because they can pair up inside it, or when it touches the **boundary**, which can absorb any leftover defect. Happy blots stop growing.
4. **Peel.** Inside each blot, pick a tree that connects its nodes. Repeatedly snip off a leaf: if the leaf is a defect, flip that edge into the correction and pass the defect to the leaf's parent. This is like pulling a loose thread back to the knot.

Why it's fast: each step only touches a blot's edges, and merges are nearly O(1). Why it's parallel: growth is local, so every blot could grow on its own processor at the same time. Why it's a bit worse than MWPM: blots grow in all directions equally, so they sometimes pair defects that aren't the cheapest pairing.

**Prediction exercise:** *"Will UF's threshold be higher or lower than MWPM's?"*

## Inputs / outputs
- **In:** `DecodingGraph`, syndrome `bool[D]`
- **Out:** `bool[K]` observable prediction, plus stats: `growth_rounds`, `num_clusters`

## Checklist
### Data structures
- [x] Disjoint-set arrays over D+1 nodes: `parent`, `rank` (or `size`), with path compression
- [x] Per-root cluster attributes: `parity` (defect count mod 2), `touches_boundary`
- [x] Per-edge `support` (int) and a `fully_grown` boolean
- [x] Per-root list of boundary edges (edges leaving the cluster), so growth doesn't scan all E edges

### v1: unweighted (half-edge growth)
- [x] Initialise: each node is its own set. Parity = 1 for defects. The boundary node has `touches_boundary=True`
- [x] A cluster is **active** if `parity == 1 and not touches_boundary`
- [x] Growth step: each active cluster adds +1 support to each of its boundary edges. An edge is full at support 2
- [x] Collect newly full edges, then `union` their endpoints: XOR the parities, OR the boundary flags, concatenate the boundary-edge lists
- [x] Loop until no active clusters remain
- [x] Peel: build a BFS spanning forest over **fully grown edges only**. Root the tree at the boundary node if the cluster contains it, otherwise at any node. Process nodes in reverse BFS order: if a node is still marked as a defect, add its parent edge to the correction and toggle the parent's defect flag. Ignore the boundary node's flag
- [x] Prediction = XOR of `edge_obs` over the correction edges

### v2: weighted
- [x] Integer weights: `w_int = max(1, round(edge_weight * scale))`, with `scale` chosen so the median `w_int` ≈ 10. The edge is full when `support >= w_int`
- [x] Each step, each active cluster adds +1 to each of its boundary edges. An edge between two active clusters gets +2
- [x] Optional speed-up ("event jump"): grow by `min(remaining / rate)` across all frontier edges instead of 1 unit at a time
- [x] Constructor flag `weighted: bool = True`; register both as `"uf"` (weighted) and `"uf_unweighted"`

### Tests (`test_union_find.py`)
- [x] Zero syndrome → zero prediction, zero growth rounds
- [x] 5-bit repetition-code toy graph: single errors, an adjacent pair, an error at the boundary. Hand-check the answers
- [x] **Single fault, d=3 and d=5: 100%** (both unweighted and weighted)
- [x] Agreement with PyMatching ≥ 99% at p = 0.001, d = 5 (10k shots)
- [x] Peeling invariant: syndrome of the correction == the input syndrome (check this on random shots)
- [x] `@slow`: LER_UF ≤ 3× LER_PM for d ∈ {3,5} at p = 0.003

### Results
- [x] Add `uf` and `uf_unweighted` to the sweep; regenerate the plot
- [x] Record the growth-rounds distribution vs d (parallel-depth proxy)
- [x] Log in PROGRESS.md: LER ratio to PyMatching, rough UF threshold, latency

## Acceptance criteria
1. Single-fault test passes 100%.
2. The peeling invariant holds on 1,000 random shots.
3. Weighted UF beats unweighted UF on circuit noise (it should, clearly).
4. LER_UF is within 3× of PyMatching below threshold. If it's worse than 5×, there's a bug: go back to the single-fault and toy tests.

## Agent notes
- **The single boundary node is fine.** Clusters that touch it become neutral. Several clusters may merge through it, and that's still correct, because peeling rooted at the boundary handles each branch independently.
- The most common bugs: (1) growing edges *inside* a cluster, (2) peeling over half-grown edges, (3) forgetting to toggle the parent's defect flag, (4) giving the boundary node a defect flag.
- Draw the d=3, rounds=1 graph (networkx + matplotlib) with defects and the final clusters for one shot. It's great for debugging *and* for a slide.
- Pure Python first. Only reach for numba if d=7 takes over ~2 ms/shot, and keep the pure version as the reference implementation.

## Talk material to capture
- Animated or 3-frame picture: defects → grown clusters → peeled correction.
- LER vs p: UF vs PyMatching.
- Growth rounds vs d (the parallelism story).

## Kick-off prompt
> Read CLAUDE.md and docs/phases/P1_union_find.md. Explain Union-Find to me with the ink-blot picture, then build v1 (unweighted) with its tests, including the single-fault test, before touching v2. Show me a drawing of one decoded d=3 shot. Stop when the P1 acceptance criteria pass.
