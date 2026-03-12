# Gated DeltaNet Extract (Pruned)

This extraction is a pruned Gated DeltaNet working set.

## What Changed
- Rebuilt from a strict GDN seed set instead of broad keyword collection.
- Removed unrelated model families that were previously pulled in by aggregator imports.
- Kept only files required by GDN code paths and their direct local dependencies.

## Current Size
- Total files: 118 (see `MANIFEST.txt`)
- Seed files: 28 (see `SEED_FILES.txt`)
- Added dependency files: 86 (see `ADDED_IMPORT_DEPS.txt`)

## Import Status
- Internal absolute imports (`fla/...`, `tests/...`): resolved.
- Internal relative imports: resolved.
- Validation result: `missing_abs_refs=0`, `missing_rel_refs=0`, `missing_internal_refs=0`.

## Scope Notes
- Kept: Gated DeltaNet layer/model, gated delta rule kernels, CP/common kernels, utility modules, GDN tests, and directly GDN-related benchmark/comparison files.
- Verified: no directly GDN-related file from source scanning is missing in this extraction (`missing_gdn_related_now=0`).

## Repository Usage
- Push `gated_deltanet_extract` as your new repository root.
- Keep current import style (`from fla...`) unchanged.
