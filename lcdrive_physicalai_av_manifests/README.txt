LCDrive PhysicalAI-AV Reproducibility Manifests
Generated: 2026-06-25

Contents are sanitized for sharing: only clip UUIDs, split labels, and validation scenario categories are included.

External availability check:
- All train/val UUIDs were checked against clip_index.parquet, the PhysicalAI-AV public clip index used by the LCDrive subset construction. The UUID field is the parquet index named clip_id.
- missing_from_clip_index_count = 0
- uuid_format_bad_count = 0
- train_val_overlap_count = 0
- train split mismatch count = 0
- val split mismatch count = 0
- train invalid/missing clip_is_valid count = 0
- val invalid/missing clip_is_valid count = 0

Counts:
- train unique clip UUIDs: 39072
- val unique clip UUIDs: 23758

Files:
- lcdrive_train_val_split_clip_uuids.csv: one row per clip, columns clip_uuid, split.
- lcdrive_train_clip_uuids.txt / lcdrive_val_clip_uuids.txt: plain UUID lists.
- lcdrive_val_primary_scenario_for_table2.csv: one validation row per clip; use this to group ADE by scenario for the paper Table 2-style split.
- lcdrive_val_all_scenario_rows.csv: all recovered validation scenario rows for audit; some clips have multiple recovered scenarios.
- lcdrive_val_all_scenarios_per_clip_audit.csv: one validation row per clip with all recovered scenarios joined by '|'.
- manifest_summary.json: validation counts and category distributions.

Table 2 grouping note:
The original category-analysis notebook used uuid_to_category = dict(zip(category_df['key'], category_df['odd'])). This package mirrors that behavior in lcdrive_val_primary_scenario_for_table2.csv: for duplicated validation UUID rows, the last recovered category is used as the single primary grouping label. The raw labels use "General Training/Validation"; the paper-facing column maps it to "General Driving".
