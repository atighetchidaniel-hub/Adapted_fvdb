using System;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

[RequireComponent(typeof(PVSCameraController))]
public class NeuralPVSBridge : MonoBehaviour
{
    public bool bridgeEnabled = true;
    public string pythonExecutable = "python";
    public string bridgeScript = "Tools/NeuralPVSBridge/unity_bridge_infer.py";
    public string neuralPVSRoot = "/Users/danielatighetchi/Desktop/Adapted_fvdb";
    public string checkpointPath = "";
    public string predictionFolderOverride = "";

    public string model = "OACNNsInterleaved";
    public string backend = "fvdb";
    public int classes = 1;
    public int inChannels = 1;
    public int modelDepth = 2;
    public int interleaverR = 2;
    public int zSize = 256;
    public float threshold = 0.5f;
    public int maxPoolSize = -1;
    public string device = "";

    public int pollEveryFrames = 5;
    public float predictionTimeoutSeconds = 120.0f;
    public bool loadExistingPrediction = true;
    public bool logProcessOutput = true;

    private PVSCameraController controller;
    private Task<BridgeResult> runningTask;
    private int runningViewCell = -1;
    private int loadedViewCell = -1;

    private struct BridgeResult
    {
        public bool success;
        public int viewCellIndex;
        public string outputPath;
        public string stdout;
        public string stderr;
        public string error;
    }

    private void Awake()
    {
        controller = GetComponent<PVSCameraController>();
    }

    private void Update()
    {
        if (!bridgeEnabled || controller == null)
            return;

        CompleteFinishedPrediction();

        if (Time.frameCount % Mathf.Max(1, pollEveryFrames) != 0)
            return;

        if (controller.mode != PVSMode.GenerateGVAndRenderNeuralPVV)
            return;

        int viewCellIndex = ViewCell.InstanceCount - 1;
        if (viewCellIndex < 0)
            return;

        string gvPath = Path.Combine(controller.GvFolder, $"{viewCellIndex}_gv.bin.gz");
        string outputPath = GetPredictionPath(viewCellIndex);

        if (loadExistingPrediction && loadedViewCell != viewCellIndex && File.Exists(outputPath))
        {
            LoadPrediction(outputPath, viewCellIndex);
            return;
        }

        if (runningTask != null || runningViewCell == viewCellIndex)
            return;

        if (!File.Exists(gvPath) || string.IsNullOrWhiteSpace(checkpointPath))
            return;

        StartPrediction(gvPath, outputPath, viewCellIndex);
    }

    private void CompleteFinishedPrediction()
    {
        if (runningTask == null || !runningTask.IsCompleted)
            return;

        BridgeResult result = runningTask.Result;
        runningTask = null;
        runningViewCell = -1;

        if (logProcessOutput && !string.IsNullOrWhiteSpace(result.stdout))
            UnityEngine.Debug.Log(result.stdout);
        if (!string.IsNullOrWhiteSpace(result.stderr))
            UnityEngine.Debug.LogWarning(result.stderr);

        if (!result.success)
        {
            UnityEngine.Debug.LogError($"NeuralPVS prediction failed for view cell {result.viewCellIndex}: {result.error}");
            return;
        }

        LoadPrediction(result.outputPath, result.viewCellIndex);
    }

    private void StartPrediction(string gvPath, string outputPath, int viewCellIndex)
    {
        string scriptPath = ResolveProjectPath(bridgeScript);
        string neuralRoot = ResolvePath(neuralPVSRoot);
        string checkpoint = ResolvePath(checkpointPath);

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath));

        runningViewCell = viewCellIndex;
        runningTask = Task.Run(() => RunPredictionProcess(
            viewCellIndex,
            scriptPath,
            neuralRoot,
            gvPath,
            outputPath,
            checkpoint
        ));

        UnityEngine.Debug.Log($"Started NeuralPVS prediction for view cell {viewCellIndex}: {gvPath}");
    }

    private BridgeResult RunPredictionProcess(
        int viewCellIndex,
        string scriptPath,
        string neuralRoot,
        string gvPath,
        string outputPath,
        string checkpoint)
    {
        try
        {
            var args = new StringBuilder();
            AppendArg(args, scriptPath);
            AppendArg(args, "--neuralpvs-root");
            AppendArg(args, neuralRoot);
            AppendArg(args, "--gv");
            AppendArg(args, gvPath);
            AppendArg(args, "--out");
            AppendArg(args, outputPath);
            AppendArg(args, "--checkpoint");
            AppendArg(args, checkpoint);
            AppendArg(args, "--model");
            AppendArg(args, model);
            AppendArg(args, "--backend");
            AppendArg(args, backend);
            AppendArg(args, "--classes");
            AppendArg(args, classes.ToString());
            AppendArg(args, "--in-channels");
            AppendArg(args, inChannels.ToString());
            AppendArg(args, "--model-depth");
            AppendArg(args, modelDepth.ToString());
            AppendArg(args, "--interleaver-r");
            AppendArg(args, interleaverR.ToString());
            AppendArg(args, "--z-size");
            AppendArg(args, zSize.ToString());
            AppendArg(args, "--threshold");
            AppendArg(args, threshold.ToString(System.Globalization.CultureInfo.InvariantCulture));
            AppendArg(args, "--max-pool-size");
            AppendArg(args, maxPoolSize.ToString());
            if (!string.IsNullOrWhiteSpace(device))
            {
                AppendArg(args, "--device");
                AppendArg(args, device);
            }

            var processInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = pythonExecutable,
                Arguments = args.ToString(),
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                WorkingDirectory = ResolveProjectPath("."),
            };

            using (var process = System.Diagnostics.Process.Start(processInfo))
            {
                if (process == null)
                    throw new InvalidOperationException("Failed to start Python process.");

                bool finished = process.WaitForExit((int)(predictionTimeoutSeconds * 1000.0f));
                if (!finished)
                {
                    process.Kill();
                    string timedOutStdout = process.StandardOutput.ReadToEnd();
                    string timedOutStderr = process.StandardError.ReadToEnd();
                    return Fail(viewCellIndex, outputPath, timedOutStdout, timedOutStderr, "Timed out.");
                }

                string stdout = process.StandardOutput.ReadToEnd();
                string stderr = process.StandardError.ReadToEnd();

                if (process.ExitCode != 0)
                    return Fail(viewCellIndex, outputPath, stdout, stderr, $"Exit code {process.ExitCode}.");

                if (!File.Exists(outputPath))
                    return Fail(viewCellIndex, outputPath, stdout, stderr, "Prediction process finished but output file is missing.");

                return new BridgeResult
                {
                    success = true,
                    viewCellIndex = viewCellIndex,
                    outputPath = outputPath,
                    stdout = stdout,
                    stderr = stderr,
                    error = "",
                };
            }
        }
        catch (Exception ex)
        {
            return Fail(viewCellIndex, outputPath, "", "", ex.ToString());
        }
    }

    private BridgeResult Fail(int viewCellIndex, string outputPath, string stdout, string stderr, string error)
    {
        return new BridgeResult
        {
            success = false,
            viewCellIndex = viewCellIndex,
            outputPath = outputPath,
            stdout = stdout,
            stderr = stderr,
            error = error,
        };
    }

    private void LoadPrediction(string outputPath, int viewCellIndex)
    {
        Texture3D pvvTexture = Utils.LoadPVV(
            outputPath,
            controller.volumeSettings.volumeSize,
            controller.volumeSettings.volumeSize,
            controller.volumeSettings.volumeDepth
        );

        if (pvvTexture == null)
        {
            UnityEngine.Debug.LogWarning($"Failed to load NeuralPVS prediction: {outputPath}");
            return;
        }

        controller.SetNeuralPVV(pvvTexture, outputPath, viewCellIndex);
        loadedViewCell = viewCellIndex;
    }

    private string GetPredictionPath(int viewCellIndex)
    {
        string folder = string.IsNullOrWhiteSpace(predictionFolderOverride)
            ? controller.pvvLoadFolder
            : ResolvePath(predictionFolderOverride);
        return Path.Combine(folder, $"{viewCellIndex}_predicted_pvv.bin.gz");
    }

    private string ResolveProjectPath(string path)
    {
        if (Path.IsPathRooted(path))
            return path;
        return Path.GetFullPath(Path.Combine(Application.dataPath, "..", path));
    }

    private string ResolvePath(string path)
    {
        if (string.IsNullOrWhiteSpace(path) || Path.IsPathRooted(path))
            return path;
        return ResolveProjectPath(path);
    }

    private static void AppendArg(StringBuilder builder, string value)
    {
        if (builder.Length > 0)
            builder.Append(' ');
        builder.Append('"');
        builder.Append(value.Replace("\"", "\\\""));
        builder.Append('"');
    }
}
