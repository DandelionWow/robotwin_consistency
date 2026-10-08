# Returned matched-collection audit

Status date: 2026-10-08

The transferred directory
`/data1/liuwenhao/Projects/robotwin_consistency_aba366` is preserved unchanged
as diagnostic evidence. All 48 entries in its `SHA256SUMS` file verify, and
its six Expert/Pi0.5 pairs have matching initial-state fingerprints. It is not
eligible as the formal Experiment 1 dataset for two independent reasons.

## Exact findings

| Pair | Expert frames | Pi0.5 frames | Expert success | Pi0.5 success |
|---|---:|---:|---|---|
| `blocks_ranking_size__seed200002` | 256 | 3071 | true | false |
| `blocks_ranking_size__seed200003` | 241 | 3208 | true | false |
| `hanging_mug__seed200001` | 183 | 2028 | true | false |
| `hanging_mug__seed200003` | 191 | 1870 | true | false |
| `stamp_seal__seed200004` | 77 | 930 | true | false |
| `stamp_seal__seed200005` | 75 | 502 | true | true |

Both `stamp_seal` pairs fail the fixed 81-frame window requirement because the
Expert side is short.

The old collector also appended the rollout's final frame even when it was not
on the 25-physics-step sampling boundary. Eleven of twelve trajectories
therefore have exactly one irregular terminal interval despite declaring a
constant `raw_frame_dt=0.1` and being encoded at 10 FPS. The only trajectory
without that extra interval is `hanging_mug__seed200001/expert`.

## Disposition

- Do not edit, pad, truncate, or delete the returned directory.
- Do not use it for formal pairwise results.
- The collector no longer appends a terminal off-grid frame and now validates
  the actual timestamp array interval by interval.
- Either side below 81 frames is now a hard rejection retained under
  `.staging/<pair_id>/rejection.json`.
- Both `stamp_seal` seeds must be replaced using corrected Expert-only
  screening before a new unified six-pair collection is started.
