# NeuralPVS Unity Bridge Bundle

This folder stores the Unity-side live-demo bridge together with the Python
single-viewcell inference script.

Copy these files into a Unity NeuralPVS project when setting up the live demo:

- `Assets/NeuralPVS/NeuralPVSBridge.cs`
- `Assets/NeuralPVS/PVSCameraController.cs`
- `Tools/NeuralPVSBridge/unity_bridge_infer.py`
- `Tools/NeuralPVSBridge/README.md`

The copied `PVSCameraController.cs` includes the extra
`GenerateGVAndRenderNeuralPVV` mode required by `NeuralPVSBridge.cs`.

In Unity, add `NeuralPVSBridge` to the same camera that has
`PVSCameraController`, then set:

- `PVSCameraController.mode` to `GenerateGVAndRenderNeuralPVV`
- `NeuralPVSBridge.pythonExecutable` to the Python executable in the fVDB env
- `NeuralPVSBridge.neuralPVSRoot` to this adapted fVDB repo
- `NeuralPVSBridge.checkpointPath` to a trained `.pth` checkpoint
- model settings to match the checkpoint's `training_arguments.json`

The bridge flow is:

1. Unity exports `gv/<id>_gv.bin.gz`.
2. `NeuralPVSBridge.cs` starts `unity_bridge_infer.py`.
3. Python writes `<id>_predicted_pvv.bin.gz`.
4. Unity loads the PVV and binds it to `_PVV` for the visibility-aware shader.
