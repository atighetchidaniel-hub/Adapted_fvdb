# fVDB Migration Summary

This document summarizes the major changes made to migrate the sparse backend from `spconv` to `fVDB` in the adapted NeuralPVS implementation.

## 1. Core Architectural Shift

The biggest conceptual change was moving away from `spconv.SparseConvTensor` as the central sparse data structure.

The `fVDB` backend uses three separate concepts:

- `fvdb.GridBatch` for sparse voxel topology
- `fvdb.JaggedTensor` for sparse feature storage
- `fvdb.ConvolutionPlan` for sparse convolution mapping

To make that fit the existing model code, a local wrapper `FvdbTensor` was introduced. It bundles:

- `grid`
- `data`

This became the repo-level replacement for the role that `spconv.SparseConvTensor` previously played.

## 2. Tensor Conversion Layer

The sparse conversion utilities were rewritten around `fVDB`.

Main changes in `utils/tensor.py`:

- Added `FvdbTensor`
- Added dense-to-fVDB conversion using:
  - `fvdb.JaggedTensor.from_list_of_tensors(...)`
  - `fvdb.GridBatch.from_ijk(...)`
  - `grid.inject_from_dense_cmajor(...)`
- Added fVDB-to-dense conversion using:
  - `grid.inject_to_dense_cmajor(...)`
- Updated sparse helper functions like:
  - `to_sparse(...)`
  - `to_dense(...)`
  - `to_dtype(...)`
  - `requires_grad(...)`
  - `get_dtype(...)`

This is what enabled the rest of the models and losses to operate on `fVDB` data instead of `spconv`.

## 3. VNet Backend Rewrite

The VNet sparse backend was reimplemented in terms of `fVDB`.

Main changes in `backends/fvdb.py`:

- Replaced `spconv` sparse layers with `fvdb.nn` layers
- Implemented same-grid sparse blocks using:
  - `fvdb.nn.SparseConv3d`
  - `fvdb.nn.BatchNorm`
  - `fvdb.ConvolutionPlan.from_grid_batch(...)`
- Implemented downsampling using:
  - `GridBatch.coarsened_grid(...)`
  - `GridBatch.max_pool(...)`
- Implemented upsampling using:
  - `GridBatch.refine(...)`
- Kept model-stage interfaces compatible with existing VNet code by returning `FvdbTensor`

This part was closely informed by the official `fvdb.nn.simple_unet` implementation and the `SimpleUNetBasicBlock`, `SimpleUNetDown`, and `SimpleUNetUp` patterns from the fVDB docs.

## 4. Interleaver / Deinterleaver Migration

The sparse interleaving path was ported from coordinate-feature logic written for `spconv` to `fVDB`.

Main changes in `modules/interleaver.py`:

- Replaced `spconv` sparse handling with `FvdbTensor` handling
- Used:
  - `grid.ijk`
  - `data.jdata`
  - `fvdb.JaggedTensor.from_list_of_tensors(...)`
  - `fvdb.GridBatch.from_ijk(...)`
  - `grid.inject_from_ijk(...)`
- Preserved dense interleaving and deinterleaving behavior
- Added sparse interleaving/deinterleaving support for `FvdbTensor`

This was essential for `VNetInterleaved` and `OACNNsInterleaved`.

## 5. OA-CNN Migration

The OA-CNN implementation was ported to `fVDB`.

Main changes in `models/oacnn.py`:

- Replaced the sparse backbone implementation with `fVDB` blocks
- Added `FvdbPointwise` for pointwise channel mixing
- Added `FvdbConvNormAct`
- Added `FvdbBasicBlock`
- Added `FvdbDownBlock`
- Added `FvdbUpBlock`
- Added `FvdbStem`
- Converted OA-style dict input into `FvdbTensor`

Important implementation detail:

- Downsampling was rewritten with:
  - `GridBatch.coarsened_grid(...)`
  - `GridBatch.max_pool(...)`
- Upsampling was rewritten with:
  - `GridBatch.refine(...)`
- Same-grid sparse convolution paths were rewritten with:
  - `fvdb.nn.SparseConv3d`
  - `fvdb.nn.BatchNorm`
  - `ConvolutionPlan.from_grid_batch(...)`

This allowed both:

- `OACNNs`
- `OACNNsInterleaved`

to work with `backend="fvdb"`.

## 6. Loss Migration

The sparse Dice loss was updated to work directly with `FvdbTensor`.

Main changes in `losses/dice.py`:

- Replaced sparse target gathering based on `spconv` coordinates
- Added fVDB-native target alignment using:
  - `input.grid.inject_from_dense_cmajor(target)`
- Computed sparse Dice on:
  - `input.data.jdata`

This allowed sparse supervision to remain sparse instead of densifying everything first.

## 7. Metrics, Visualization, and Runner Plumbing

Several support modules were updated so the `fVDB` path behaved like a first-class backend.

Main changes:

- `losses/metrics.py`
  - metrics now work cleanly with outputs that are converted through `to_dense(...)`
  - fixed a `gv`-related assumption in validation metrics

- `utils/vis.py`
  - visualization utilities understand `FvdbTensor`

- `modules/runner.py`
  - timing, train, validate, and infer flows now use `to_sparse(..., "fvdb")`
  - backend timing reflects the actual sparse input path

- `utils/init.py`
  - model initialization supports `fvdb`
  - sparse Dice routes correctly to the sparse loss

- `utils/args.py`
  - CLI/backend argument handling was updated for `fvdb`

## 8. Documentation and Dependency Direction

The repo was moved toward `fVDB` as the sparse backend direction.

This included changes such as:

- updating backend registration
- updating test scripts
- updating README/backend examples
- moving dependency assumptions away from `spconv`

## 9. Main fVDB APIs Used

The migration relied primarily on these official `fVDB` APIs:

- `fvdb.GridBatch`
- `fvdb.JaggedTensor`
- `fvdb.ConvolutionPlan.from_grid_batch(...)`
- `fvdb.GridBatch.from_ijk(...)`
- `fvdb.GridBatch.inject_from_dense_cmajor(...)`
- `fvdb.GridBatch.inject_to_dense_cmajor(...)`
- `fvdb.GridBatch.max_pool(...)`
- `fvdb.GridBatch.coarsened_grid(...)`
- `fvdb.GridBatch.refine(...)`
- `fvdb.nn.SparseConv3d`
- `fvdb.nn.BatchNorm`

These APIs are consistent with the official `fVDB` documentation and the `SimpleUNet` reference implementation.

## 10. Short Conversion Summary

The migration was not just a one-to-one operator swap.

It was a structural rewrite from:

- a single sparse tensor abstraction (`spconv.SparseConvTensor`)

to:

- explicit sparse topology (`GridBatch`)
- explicit sparse features (`JaggedTensor`)
- explicit sparse convolution planning (`ConvolutionPlan`)

That change affected:

- tensor conversion
- model backends
- interleaving
- OA-CNN blocks
- sparse loss computation
- runner utilities

In short, the project moved from a `spconv`-centric sparse design to an `fVDB`-native sparse design.
