# Status Update - 2026-04-09

## Summary
The `spconv` to `fvdb` backend migration in the adapted NeuralPVS repo is now implemented and validated on both synthetic data and real Unity-generated data.

The adapted repo is now `fvdb`-only and supports the major model families used in NeuralPVS.

## Main Implementation Progress
The following parts of the repo were migrated from the old sparse backend design to `fvdb`:

- sparse tensor representation
- VNet backend
- interleaver / deinterleaver
- OA-CNN models
- sparse Dice handling
- runner / utility plumbing
- CLI / model initialization support

Conceptually, the sparse backend changed from a `spconv`-style single sparse tensor abstraction to an `fvdb`-native design based on:

- `fvdb.GridBatch`
- `fvdb.JaggedTensor`
- `fvdb.ConvolutionPlan`

## Models Currently Working with `fvdb`
The following model variants now work with the `fvdb` backend:

- VNet
- VNetInterleaved
- VNetLighter
- VNetLight
- OACNNs
- OACNNsInterleaved

## Validation Completed So Far

### 1. Synthetic validation
Completed:
- model construction
- forward pass
- loss computation
- backward pass
- optimizer step
- runner-level smoke test

### 2. Real Unity-generated data validation
A real dataset was generated from Unity and successfully consumed by the adapted Python repo.

#### SceneGeneration dataset
A first Unity-generated dataset was exported and used to verify:
- dataset loading
- real-data forward pass
- full `train.py` execution on real exported data

#### Viking dataset
A second real dataset was generated from the Viking Village scene.

Dataset properties:
- dataset root: `data_for_test/viking/r30`
- samples: `818`
- sample shape: `(1, 256, 256, 256)`

Loader sanity check:
- input sum: `423489.0`
- target sum: `112749.0`

This confirmed that the Unity-exported `gv` and `pvv` data is valid and readable by the Python training code.

## Real-Data Viking Results

### Forward-pass sweep on real Unity data
All major models were run on the real Viking dataset and produced finite outputs with correct shape.

- VNet: Dice `0.8029316`
- VNetInterleaved: Dice `0.73332864`
- VNetLighter: Dice `0.52646494`
- VNetLight: Dice `0.7894346`
- OACNNs: Dice `0.77284336`
- OACNNsInterleaved: Dice `0.73182225`

All returned:
- output type: `FvdbTensor`
- output shape: `(1, 1, 256, 256, 256)`
- finite outputs: `True`

### Train-step sweep on real Unity data
All major models also completed a real-data train step successfully.

- VNet: grad_ok = `True`, Dice `0.7911557`
- VNetInterleaved: grad_ok = `True`, Dice `0.72633153`
- VNetLighter: grad_ok = `True`, Dice `0.5635845`
- VNetLight: grad_ok = `True`, Dice `0.70177466`
- OACNNs: grad_ok = `True`, Dice `0.8083265`
- OACNNsInterleaved: grad_ok = `True`, Dice `0.72402525`

This confirms:
- forward works on real data
- loss works on real data
- backward works on real data
- optimizer step works on real data

## Current Conclusion
The `fvdb` migration is no longer only an implementation prototype.

It is now validated across:
- all major model families
- synthetic smoke tests
- real Unity-generated data
- forward and train-step execution

In practical terms, the neural-network refactor to `fvdb` is working.

## Remaining Work
The next phase is no longer sparse-backend implementation. The remaining work is:

- cleanup / packaging of the adapted repo
- training real checkpoints for selected models
- running inference to generate predicted PVV outputs
- comparing predicted PVV against Unity ground-truth PVV
- integrating predicted PVV back into the Unity rendering pipeline
- building a demo loop for Unity + neural inference

## Recommended Next Steps
1. Clean and package the repo for presentation and reproducibility.
2. Train one or two proper checkpoints, likely:
   - VNet
   - OACNNsInterleaved
3. Run inference to produce `*_predicted_pvv.bin.gz`.
4. Compare predicted PVV against Unity-generated ground truth.
5. Feed predicted PVV back into Unity using the existing `RenderPVV` path.
