# Legacy Pi0.5 data provenance audit

## Formal decision

```text
usable for matched experiment: NO
```

No direct artifact or reachable Git history links `seed.txt[i]` to `episode{i}` for this
legacy dataset. File order and timestamps are circumstantial evidence only. All 500 episodes
are therefore excluded from formal matched-pair statistics.

## Complete inventory

- source: `/data1/liuwenhao/Datasets/data`
- tasks: 50
- episodes: 500
- task seed-token counts: `{10: 50}`
- task unique-seed counts: `{10: 50}`
- HDF5 head-RGB length range: 401–1701
- video FPS values: `[30.0]`
- video geometry values: `[(640, 480)]`
- Pickle key schemas: `{'["actions"]': 500}`
- unexpected/hidden sidecar files: 0
- machine-readable per-episode audit: `/data1/liuwenhao/Projects/robotwin_consistency/experiments/policy_shift/provenance/pi05_old_episode_audit.jsonl`

## Direct-evidence search

The audit checked all 50 `seed.txt` files, all 500 HDF5 files and their attributes,
all 500 trajectory Pickles, all 500 videos, `_result.txt`, unexpected/hidden sidecars,
the current RoboTwin scripts, all locally reachable RoboTwin Git branches/history, and
available shell history. Findings:

- no `_fail.mp4` writer exists in reachable RoboTwin Git history;
- current and historical eval code append `suc_test_seed_list` but do not persist it;
- `collect_data.py` writes expert collection seeds, but its trajectory schema is not the
  legacy Pi0.5 `{'actions': ...}` schema;
- no scene info, launch log, checkpoint identity, exact task-config bytes, or episode-level
  success metadata is stored beside the legacy episodes;
- filesystem timestamps are retained in JSONL only as auxiliary evidence and were not used
  to upgrade any status.

## Status interpretation

- `seed_mapping_status=AMBIGUOUS`: a positional candidate exists, but no direct writer or
  immutable manifest proves the association.
- `failure_label_status=AMBIGUOUS`: `_fail` is only a filename label; task-level result files
  do not establish an episode-level ground-truth label.
- `policy_checkpoint_status=UNKNOWN`: no policy checkpoint/config hash is archived.
- `task_config_status=AMBIGUOUS`: `demo_clean` is a directory label, not preserved config bytes.

## Per-episode decision

| task | episode | positional seed | seed mapping | failure label | policy checkpoint | task config | formal pair |
|---|---:|---:|---|---|---|---|---|
| adjust_bottle | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| adjust_bottle | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| adjust_bottle | 2 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| adjust_bottle | 3 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| adjust_bottle | 4 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| adjust_bottle | 5 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| adjust_bottle | 6 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| adjust_bottle | 7 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| adjust_bottle | 8 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| adjust_bottle | 9 | 200010 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| beat_block_hammer | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| beat_block_hammer | 1 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| beat_block_hammer | 2 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| beat_block_hammer | 3 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| beat_block_hammer | 4 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| beat_block_hammer | 5 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| beat_block_hammer | 6 | 200010 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| beat_block_hammer | 7 | 200011 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| beat_block_hammer | 8 | 200014 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| beat_block_hammer | 9 | 200015 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| blocks_ranking_rgb | 0 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| blocks_ranking_rgb | 1 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| blocks_ranking_rgb | 2 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| blocks_ranking_rgb | 3 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| blocks_ranking_rgb | 4 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| blocks_ranking_rgb | 5 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| blocks_ranking_rgb | 6 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| blocks_ranking_rgb | 7 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| blocks_ranking_rgb | 8 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| blocks_ranking_rgb | 9 | 200010 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| blocks_ranking_size | 0 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| blocks_ranking_size | 1 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| blocks_ranking_size | 2 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| blocks_ranking_size | 3 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| blocks_ranking_size | 4 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| blocks_ranking_size | 5 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| blocks_ranking_size | 6 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| blocks_ranking_size | 7 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| blocks_ranking_size | 8 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| blocks_ranking_size | 9 | 200011 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| click_alarmclock | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| click_alarmclock | 1 | 300000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| click_alarmclock | 2 | 300001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| click_alarmclock | 3 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| click_alarmclock | 4 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| click_alarmclock | 5 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| click_alarmclock | 6 | 300002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| click_alarmclock | 7 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| click_alarmclock | 8 | 300006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| click_alarmclock | 9 | 200010 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| click_bell | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| click_bell | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| click_bell | 2 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| click_bell | 3 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| click_bell | 4 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| click_bell | 5 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| click_bell | 6 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| click_bell | 7 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| click_bell | 8 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| click_bell | 9 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| dump_bin_bigbin | 0 | 300004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| dump_bin_bigbin | 1 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| dump_bin_bigbin | 2 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| dump_bin_bigbin | 3 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| dump_bin_bigbin | 4 | 300005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| dump_bin_bigbin | 5 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| dump_bin_bigbin | 6 | 200013 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| dump_bin_bigbin | 7 | 200014 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| dump_bin_bigbin | 8 | 200015 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| dump_bin_bigbin | 9 | 200016 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| grab_roller | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| grab_roller | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| grab_roller | 2 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| grab_roller | 3 | 300002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| grab_roller | 4 | 300003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| grab_roller | 5 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| grab_roller | 6 | 300005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| grab_roller | 7 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| grab_roller | 8 | 300006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| grab_roller | 9 | 300007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| handover_block | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| handover_block | 1 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| handover_block | 2 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| handover_block | 3 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| handover_block | 4 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| handover_block | 5 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| handover_block | 6 | 200010 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| handover_block | 7 | 200011 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| handover_block | 8 | 200013 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| handover_block | 9 | 200014 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| handover_mic | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| handover_mic | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| handover_mic | 2 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| handover_mic | 3 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| handover_mic | 4 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| handover_mic | 5 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| handover_mic | 6 | 200010 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| handover_mic | 7 | 200013 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| handover_mic | 8 | 200014 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| handover_mic | 9 | 200015 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| hanging_mug | 0 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| hanging_mug | 1 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| hanging_mug | 2 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| hanging_mug | 3 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| hanging_mug | 4 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| hanging_mug | 5 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| hanging_mug | 6 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| hanging_mug | 7 | 200010 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| hanging_mug | 8 | 200011 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| hanging_mug | 9 | 200013 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| lift_pot | 0 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| lift_pot | 1 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| lift_pot | 2 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| lift_pot | 3 | 200012 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| lift_pot | 4 | 200014 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| lift_pot | 5 | 200015 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| lift_pot | 6 | 200016 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| lift_pot | 7 | 200020 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| lift_pot | 8 | 200024 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| lift_pot | 9 | 200026 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_can_pot | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_can_pot | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_can_pot | 2 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_can_pot | 3 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_can_pot | 4 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_can_pot | 5 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_can_pot | 6 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_can_pot | 7 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_can_pot | 8 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_can_pot | 9 | 200010 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_pillbottle_pad | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_pillbottle_pad | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_pillbottle_pad | 2 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_pillbottle_pad | 3 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_pillbottle_pad | 4 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_pillbottle_pad | 5 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_pillbottle_pad | 6 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_pillbottle_pad | 7 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_pillbottle_pad | 8 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_pillbottle_pad | 9 | 200011 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_playingcard_away | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_playingcard_away | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_playingcard_away | 2 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_playingcard_away | 3 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_playingcard_away | 4 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_playingcard_away | 5 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_playingcard_away | 6 | 300000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_playingcard_away | 7 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_playingcard_away | 8 | 300001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_playingcard_away | 9 | 200010 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_stapler_pad | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_stapler_pad | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_stapler_pad | 2 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_stapler_pad | 3 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_stapler_pad | 4 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_stapler_pad | 5 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_stapler_pad | 6 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_stapler_pad | 7 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_stapler_pad | 8 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| move_stapler_pad | 9 | 200010 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| open_laptop | 0 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| open_laptop | 1 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| open_laptop | 2 | 300000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| open_laptop | 3 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| open_laptop | 4 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| open_laptop | 5 | 300002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| open_laptop | 6 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| open_laptop | 7 | 300003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| open_laptop | 8 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| open_laptop | 9 | 200010 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| open_microwave | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| open_microwave | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| open_microwave | 2 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| open_microwave | 3 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| open_microwave | 4 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| open_microwave | 5 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| open_microwave | 6 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| open_microwave | 7 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| open_microwave | 8 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| open_microwave | 9 | 200010 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| pick_diverse_bottles | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| pick_diverse_bottles | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| pick_diverse_bottles | 2 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| pick_diverse_bottles | 3 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| pick_diverse_bottles | 4 | 200011 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| pick_diverse_bottles | 5 | 200012 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| pick_diverse_bottles | 6 | 200013 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| pick_diverse_bottles | 7 | 200017 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| pick_diverse_bottles | 8 | 200020 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| pick_diverse_bottles | 9 | 200022 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| pick_dual_bottles | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| pick_dual_bottles | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| pick_dual_bottles | 2 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| pick_dual_bottles | 3 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| pick_dual_bottles | 4 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| pick_dual_bottles | 5 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| pick_dual_bottles | 6 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| pick_dual_bottles | 7 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| pick_dual_bottles | 8 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| pick_dual_bottles | 9 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_a2b_left | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_a2b_left | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_a2b_left | 2 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_a2b_left | 3 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_a2b_left | 4 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_a2b_left | 5 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_a2b_left | 6 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_a2b_left | 7 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_a2b_left | 8 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_a2b_left | 9 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_a2b_right | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_a2b_right | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_a2b_right | 2 | 300001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_a2b_right | 3 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_a2b_right | 4 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_a2b_right | 5 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_a2b_right | 6 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_a2b_right | 7 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_a2b_right | 8 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_a2b_right | 9 | 200010 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_bread_basket | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_bread_basket | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_bread_basket | 2 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_bread_basket | 3 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_bread_basket | 4 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_bread_basket | 5 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_bread_basket | 6 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_bread_basket | 7 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_bread_basket | 8 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_bread_basket | 9 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_bread_skillet | 0 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_bread_skillet | 1 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_bread_skillet | 2 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_bread_skillet | 3 | 200011 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_bread_skillet | 4 | 200019 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_bread_skillet | 5 | 200026 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_bread_skillet | 6 | 200027 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_bread_skillet | 7 | 200030 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_bread_skillet | 8 | 200035 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_bread_skillet | 9 | 200037 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_burger_fries | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_burger_fries | 1 | 300000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_burger_fries | 2 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_burger_fries | 3 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_burger_fries | 4 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_burger_fries | 5 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_burger_fries | 6 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_burger_fries | 7 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_burger_fries | 8 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_burger_fries | 9 | 300001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_can_basket | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_can_basket | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_can_basket | 2 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_can_basket | 3 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_can_basket | 4 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_can_basket | 5 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_can_basket | 6 | 200012 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_can_basket | 7 | 200013 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_can_basket | 8 | 200014 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_can_basket | 9 | 200016 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_cans_plasticbox | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_cans_plasticbox | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_cans_plasticbox | 2 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_cans_plasticbox | 3 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_cans_plasticbox | 4 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_cans_plasticbox | 5 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_cans_plasticbox | 6 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_cans_plasticbox | 7 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_cans_plasticbox | 8 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_cans_plasticbox | 9 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_container_plate | 0 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_container_plate | 1 | 200013 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_container_plate | 2 | 200014 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_container_plate | 3 | 200017 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_container_plate | 4 | 200018 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_container_plate | 5 | 200021 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_container_plate | 6 | 200023 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_container_plate | 7 | 200025 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_container_plate | 8 | 200026 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_container_plate | 9 | 200028 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_dual_shoes | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_dual_shoes | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_dual_shoes | 2 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_dual_shoes | 3 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_dual_shoes | 4 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_dual_shoes | 5 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_dual_shoes | 6 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_dual_shoes | 7 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_dual_shoes | 8 | 200010 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_dual_shoes | 9 | 200011 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_empty_cup | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_empty_cup | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_empty_cup | 2 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_empty_cup | 3 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_empty_cup | 4 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_empty_cup | 5 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_empty_cup | 6 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_empty_cup | 7 | 200010 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_empty_cup | 8 | 200011 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_empty_cup | 9 | 200012 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_fan | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_fan | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_fan | 2 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_fan | 3 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_fan | 4 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_fan | 5 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_fan | 6 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_fan | 7 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_fan | 8 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_fan | 9 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_mouse_pad | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_mouse_pad | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_mouse_pad | 2 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_mouse_pad | 3 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_mouse_pad | 4 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_mouse_pad | 5 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_mouse_pad | 6 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_mouse_pad | 7 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_mouse_pad | 8 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_mouse_pad | 9 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_basket | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_basket | 1 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_basket | 2 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_basket | 3 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_basket | 4 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_basket | 5 | 200014 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_basket | 6 | 200015 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_basket | 7 | 200016 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_basket | 8 | 200017 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_basket | 9 | 200018 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_scale | 0 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_scale | 1 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_scale | 2 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_scale | 3 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_scale | 4 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_scale | 5 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_scale | 6 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_scale | 7 | 200011 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_scale | 8 | 200015 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_scale | 9 | 200016 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_stand | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_stand | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_stand | 2 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_stand | 3 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_stand | 4 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_stand | 5 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_stand | 6 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_stand | 7 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_stand | 8 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_object_stand | 9 | 200010 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_phone_stand | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_phone_stand | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_phone_stand | 2 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_phone_stand | 3 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_phone_stand | 4 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_phone_stand | 5 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_phone_stand | 6 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_phone_stand | 7 | 200010 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_phone_stand | 8 | 200011 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_phone_stand | 9 | 200013 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_shoe | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_shoe | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_shoe | 2 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_shoe | 3 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_shoe | 4 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_shoe | 5 | 200010 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_shoe | 6 | 200012 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_shoe | 7 | 200013 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_shoe | 8 | 200014 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| place_shoe | 9 | 200015 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| press_stapler | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| press_stapler | 1 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| press_stapler | 2 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| press_stapler | 3 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| press_stapler | 4 | 200011 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| press_stapler | 5 | 200012 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| press_stapler | 6 | 200014 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| press_stapler | 7 | 200016 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| press_stapler | 8 | 200017 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| press_stapler | 9 | 200018 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| put_bottles_dustbin | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| put_bottles_dustbin | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| put_bottles_dustbin | 2 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| put_bottles_dustbin | 3 | 300000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| put_bottles_dustbin | 4 | 300001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| put_bottles_dustbin | 5 | 300002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| put_bottles_dustbin | 6 | 300003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| put_bottles_dustbin | 7 | 300004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| put_bottles_dustbin | 8 | 300005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| put_bottles_dustbin | 9 | 300008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| put_object_cabinet | 0 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| put_object_cabinet | 1 | 200012 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| put_object_cabinet | 2 | 200016 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| put_object_cabinet | 3 | 200025 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| put_object_cabinet | 4 | 200027 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| put_object_cabinet | 5 | 200038 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| put_object_cabinet | 6 | 200044 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| put_object_cabinet | 7 | 200054 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| put_object_cabinet | 8 | 200055 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| put_object_cabinet | 9 | 200067 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| rotate_qrcode | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| rotate_qrcode | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| rotate_qrcode | 2 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| rotate_qrcode | 3 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| rotate_qrcode | 4 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| rotate_qrcode | 5 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| rotate_qrcode | 6 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| rotate_qrcode | 7 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| rotate_qrcode | 8 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| rotate_qrcode | 9 | 200010 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| scan_object | 0 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| scan_object | 1 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| scan_object | 2 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| scan_object | 3 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| scan_object | 4 | 200011 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| scan_object | 5 | 300002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| scan_object | 6 | 300003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| scan_object | 7 | 300005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| scan_object | 8 | 300006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| scan_object | 9 | 300008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| shake_bottle_horizontally | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| shake_bottle_horizontally | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| shake_bottle_horizontally | 2 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| shake_bottle_horizontally | 3 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| shake_bottle_horizontally | 4 | 200015 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| shake_bottle_horizontally | 5 | 200019 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| shake_bottle_horizontally | 6 | 200024 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| shake_bottle_horizontally | 7 | 200027 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| shake_bottle_horizontally | 8 | 200033 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| shake_bottle_horizontally | 9 | 200046 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| shake_bottle | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| shake_bottle | 1 | 200013 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| shake_bottle | 2 | 200015 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| shake_bottle | 3 | 200024 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| shake_bottle | 4 | 200027 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| shake_bottle | 5 | 200031 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| shake_bottle | 6 | 200033 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| shake_bottle | 7 | 200039 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| shake_bottle | 8 | 200049 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| shake_bottle | 9 | 200051 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_blocks_three | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_blocks_three | 1 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_blocks_three | 2 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_blocks_three | 3 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_blocks_three | 4 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_blocks_three | 5 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_blocks_three | 6 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_blocks_three | 7 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_blocks_three | 8 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_blocks_three | 9 | 200010 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_blocks_two | 0 | 200001 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_blocks_two | 1 | 200002 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_blocks_two | 2 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_blocks_two | 3 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_blocks_two | 4 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_blocks_two | 5 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_blocks_two | 6 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_blocks_two | 7 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_blocks_two | 8 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_blocks_two | 9 | 200010 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_bowls_three | 0 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_bowls_three | 1 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_bowls_three | 2 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_bowls_three | 3 | 200010 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_bowls_three | 4 | 200011 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_bowls_three | 5 | 200014 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_bowls_three | 6 | 200016 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_bowls_three | 7 | 200017 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_bowls_three | 8 | 200020 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_bowls_three | 9 | 200022 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_bowls_two | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_bowls_two | 1 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_bowls_two | 2 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_bowls_two | 3 | 200012 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_bowls_two | 4 | 200014 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_bowls_two | 5 | 200015 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_bowls_two | 6 | 200016 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_bowls_two | 7 | 200019 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_bowls_two | 8 | 200020 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stack_bowls_two | 9 | 200023 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stamp_seal | 0 | 200003 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stamp_seal | 1 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stamp_seal | 2 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stamp_seal | 3 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stamp_seal | 4 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stamp_seal | 5 | 200009 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stamp_seal | 6 | 200011 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stamp_seal | 7 | 200012 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stamp_seal | 8 | 200013 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| stamp_seal | 9 | 200018 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| turn_switch | 0 | 200000 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| turn_switch | 1 | 200004 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| turn_switch | 2 | 200005 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| turn_switch | 3 | 200006 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| turn_switch | 4 | 200007 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| turn_switch | 5 | 200008 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| turn_switch | 6 | 200011 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| turn_switch | 7 | 200012 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| turn_switch | 8 | 200013 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |
| turn_switch | 9 | 200014 | AMBIGUOUS | AMBIGUOUS | UNKNOWN | AMBIGUOUS | NO |

## Required fallback

The legacy 500 episodes may remain useful as an unmatched, filename-labelled policy
reference set, but not as the core matched experiment. The next admissible path is fresh
pair-at-a-time collection with preserved policy/config/checkpoint and initial-state
fingerprints.
