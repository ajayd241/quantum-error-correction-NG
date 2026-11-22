# P4 (Stretch): Our Own Exact MWPM

**Time box:** 0:30  **Depends on:** P0  **Skip first if behind schedule**

## Goal
A small, readable exact MWPM decoder built from networkx primitives. It adds one more decoder to the count, shows we understand what PyMatching does, and gives an exactness check.

## Concept primer
MWPM asks: of all the ways to pair up the defects (or send a defect to the boundary), which pairing has the smallest total path cost? We compute the shortest path between every pair of defects, build a small complete graph of the defects, and ask a textbook algorithm (Edmonds' blossom) for the cheapest perfect pairing. It's exact but slow, roughly O(n³) in the number of defects, and it's inherently sequential. That's exactly the contrast with UF and BP.

## Checklist
- [ ] Build a weighted `nx.Graph` from `DecodingGraph` once (boundary = node D)
- [ ] Per shot: Dijkstra from each defect (`nx.single_source_dijkstra`), recording distances and paths to the other defects *and* to the boundary
- [ ] Matching graph: one node per defect plus one "boundary twin" per defect. The defect–defect edge cost is the path distance; the defect–own-twin edge cost is the distance to the boundary; twin–twin edges cost 0 (so unused twins can pair off)
- [ ] `nx.min_weight_matching` (or `max_weight_matching` on negated weights with `maxcardinality=True`)
- [ ] Prediction = XOR of `edge_obs` along the chosen paths (cache the path's observable parity during Dijkstra)
- [ ] Register `"mwpm_nx"`. Limit it to d ≤ 5 in the sweep script

### Tests (`test_mwpm_nx.py`)
- [ ] Zero syndrome; repetition-code toy cases
- [ ] Single fault: 100%
- [ ] Prediction agreement with PyMatching ≥ 99.5% at d=3, p=0.005 (ties can legitimately differ)
- [ ] `@slow`: LER statistically equal to PyMatching at d=3 (error bars overlap)

## Acceptance criteria
LER matches PyMatching within error bars at d=3. The latency table shows how much slower it is: this is the "why PyMatching exists" slide.

## Agent notes
- Don't optimize. Readability is the point.
- With 0 defects, return zeros immediately. With 1 defect, it must go to the boundary.

## Kick-off prompt
> Read CLAUDE.md and docs/phases/P4_mwpm_stretch.md. Implement the networkx MWPM decoder with the boundary-twin construction, test it against PyMatching at d=3, and add its latency to the table.
