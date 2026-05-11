# NeuralPVS Unity Bridge

This local bridge keeps the Unity renderer and the fVDB Python model loosely coupled.

The first supported flow is:

1. Unity generates a geometry volume file under `gv/<id>_gv.bin.gz`.
2. `NeuralPVSBridge.cs` starts `unity_bridge_infer.py` for that GV file.
3. Python writes `<id>_predicted_pvv.bin.gz`.
4. Unity loads the predicted PVV through the existing `Utils.LoadPVV(...)` path and binds it as `_PVV`.

The original `NeuralPVS` clone is not modified. This project copy has no `.git` folder.

## Unity Setup

1. Open this copied project in Unity.
2. Select the camera that has `PVSCameraController`.
3. Add the `NeuralPVSBridge` component.
4. Set `PVSCameraController.mode` to `GenerateGVAndRenderNeuralPVV`.
5. In `NeuralPVSBridge`, set:
   - `Python Executable`
   - `NeuralPVS Root` to your adapted fVDB repo
   - `Checkpoint Path`
   - `Model`, `Backend`, `Z Size`, and optional post-processing values

## Python Script

The bridge script is local to this Unity copy:

```text
Tools/NeuralPVSBridge/unity_bridge_infer.py
```

It imports the adapted fVDB repo from `--neuralpvs-root`, so the neural network code remains in your Python project.
