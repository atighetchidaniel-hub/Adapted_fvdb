using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Unity.Cinemachine;
using Unity.Collections;
using Unity.Profiling;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;
using UnityEngine.SceneManagement;
using static Unity.VisualScripting.Member;

public enum ExportMode
{
    ImageSequence = 0,
    Video = 1
}

public enum PVVFilter
{
    None = 1,
    Box = 2,
    Trilinear = 3
}

public enum PVSMode
{
    None = 0,
    GenerateGV = 1,
    GeneratePVV = 2,
    GenerateGVAndPVV = 3,
    RenderPVV = 4,
    GenerateGVAndRenderNeuralPVV = 5,
}

public enum PVVRenderMode
{
    FirstPerson = 1,
    ThirdPerson = 2
}

[System.Serializable]
public class VolumeSettings
{
    public int volumeSize = 256;
    public int volumeDepth = 256;
    public int samplingFactor = 2;
    public ComputeShader clearVolumeShader;
    public ViewCell viewCell = null;
    public bool linearZ = true;
    public float logDepthScale = 0.01f; // Corresponds to roughly 30m linear depth scaling with far = 500 and near 0.3
}

[RequireComponent(typeof(Camera))]
public class PVSCameraController : MonoBehaviour
{
    public UniversalRendererData pvsRenderer = null;
    private int rendererIndex = 0;

    public VolumeSettings volumeSettings;
    public GeometryVolumeFeature.GVSettings geometryVolumeSettings;
    public PotentiallyVisibleVolumeFeature.PVVSettings potentiallyVisibleVolumeSettings;

    public ViewCellSettings viewCellSettings;

    private GeometryVolumeFeature gvFeature = null;
    private PotentiallyVisibleVolumeFeature pvvFeature = null;
    private ScriptableRendererData rendererData = null;

    public PVSMode mode = PVSMode.GenerateGV;
    public PVVRenderMode pvsRenderMode = PVVRenderMode.FirstPerson;

    public int target_fps = 60;

    public int targetSeconds = 60;

    
    public List<float> radii = new List<float> { 0.3f };
    private int cur_radii_index = 0;

    public bool exportFrames = false;
     
    public ExportMode frameExportMode = ExportMode.Video;

    public string datasetPath = "./data/render/";
    public string pvvLoadFolder = "./data/render/0715-viking-r30d30-20aqPs2U";

    private string scene = "";

    private int customFrameCount = 0;

    private string gvFolder;
    private string pvvFolder;
    private string pvvColorFolder;
    private string gtColorFolder;

    private string globalTextureName = "_MainCamDepthTexture";
    private RenderTexture _customDepthTexture;

    public string GvFolder { get { return gvFolder; } }
    public string PvvFolder { get { return pvvFolder; } }
    public string PvvColorFolder { get { return pvvColorFolder; } }
    public string GtColorFolder { get { return gtColorFolder; } }

    public PVVFilter filter = PVVFilter.Trilinear;
    private bool neuralPVSHasPVV = false;
    private Texture3D neuralPVVTexture = null;
    private int neuralPVVViewCell = -1;

    private int file_count = 0;
    private bool shouldCaptureFrameThisFrame = false;  // Flag to indicate frame should be captured

    private ComputeBuffer discardCounterBuffer;
    private uint[] _countData = new uint[1];

    private bool resetViewCell = false;

    private Process ffmpeg;
    private Stream ffmpegIn;

    private bool running = true;

    private float gvTimings = 0.0f;
    private int gvSamples = 0;

    private float frameTimings = 0.0f;
    private int timingCounter = 0;

    FrameTiming[] m_FrameTimings = new FrameTiming[10];

    private void Awake()
    {
        discardCounterBuffer = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Structured, ComputeBufferMode.Dynamic);

        ResetCounter();

        Shader.SetGlobalBuffer("_DiscardCounter", discardCounterBuffer);
        
    }
    void Start()
    {
        if (pvsRenderer == null)
        {
            UnityEngine.Debug.LogError("No PVS Renderer assigned!");
            return;
        }

        if (!SystemInfo.supportsGpuRecorder)
        {
            UnityEngine.Debug.LogWarning("GPU profiling is not supported on this platform/device.");
        }

        // Ensure compute shader is assigned
        if (volumeSettings.clearVolumeShader == null)
        {
            // Try to load from Resources first
            volumeSettings.clearVolumeShader = Resources.Load<ComputeShader>("ClearGeometryVolume");
            
            // If not found in Resources, try loading directly using Unity's asset loading
            if (volumeSettings.clearVolumeShader == null)
            {
#if UNITY_EDITOR
                string[] guids = UnityEditor.AssetDatabase.FindAssets("ClearGeometryVolume t:ComputeShader");
                if (guids.Length > 0)
                {
                    string path = UnityEditor.AssetDatabase.GUIDToAssetPath(guids[0]);
                    volumeSettings.clearVolumeShader = UnityEditor.AssetDatabase.LoadAssetAtPath<ComputeShader>(path);
                    UnityEngine.Debug.Log($"Loaded ClearGeometryVolume compute shader from: {path}");
                }
#endif
            }
            
            if (volumeSettings.clearVolumeShader == null)
            {
                UnityEngine.Debug.LogError("Could not load ClearGeometryVolume compute shader. Please assign it manually in the inspector.");
                return;
            }
        }

        GeometryVolumeFeature.k_GenerateGeomSampler.enableRecording = true;

        customFrameCount = 0;
        running = true;

        viewCellSettings.radius = radii[cur_radii_index];
        cur_radii_index++;

        scene = SceneManager.GetActiveScene().name;
        var folder = $"{scene}/r{viewCellSettings.radius * 100}";
        gvFolder = Path.Combine(datasetPath, folder, "gv");
        pvvFolder = Path.Combine(datasetPath, folder, "pvv");
        pvvColorFolder = Path.Combine(pvvLoadFolder, "00_color");
        gtColorFolder = Path.Combine(datasetPath, folder, "gt_color");

        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = target_fps;
        Time.captureFramerate = target_fps;
        Time.fixedDeltaTime = 1.0f / target_fps;

        file_count = 0;

        if (mode != PVSMode.None)
        {
            System.IO.Directory.CreateDirectory(datasetPath);
            UnityEngine.Debug.Log($"Created dataset path: {Path.GetFullPath(datasetPath)}");

            if (mode == PVSMode.GenerateGV || mode == PVSMode.GenerateGVAndRenderNeuralPVV)
            {
                System.IO.Directory.CreateDirectory(gvFolder);
                UnityEngine.Debug.Log($"Created GV folder: {Path.GetFullPath(gvFolder)}");

                if (mode == PVSMode.GenerateGVAndRenderNeuralPVV)
                {
                    System.IO.Directory.CreateDirectory(pvvLoadFolder);
                    UnityEngine.Debug.Log($"Created neural PVV folder: {Path.GetFullPath(pvvLoadFolder)}");
                }
            }
            else if (mode == PVSMode.GeneratePVV)
            {
                System.IO.Directory.CreateDirectory(pvvFolder);
                UnityEngine.Debug.Log($"Created PVV folder: {Path.GetFullPath(pvvFolder)}");
            }
            else if (mode == PVSMode.GenerateGVAndPVV)
            {
                System.IO.Directory.CreateDirectory(gvFolder);
                System.IO.Directory.CreateDirectory(pvvFolder);
                UnityEngine.Debug.Log($"Created GV folder: {Path.GetFullPath(gvFolder)}");
                UnityEngine.Debug.Log($"Created PVV folder: {Path.GetFullPath(pvvFolder)}");
            }
        }

        if (exportFrames)
        {
            if (mode == PVSMode.RenderPVV || mode == PVSMode.GenerateGVAndRenderNeuralPVV)
            {
                if (Directory.Exists(pvvLoadFolder))
                    System.IO.Directory.CreateDirectory(pvvColorFolder);
                else
                {
                    UnityEngine.Debug.LogError("PVV Load Folder not found!");
#if UNITY_EDITOR
                    UnityEditor.EditorApplication.isPlaying = false;
#else
                    Application.Quit();
#endif
                }
            }
            else
                System.IO.Directory.CreateDirectory(gtColorFolder);
        }

        var urpAsset = GraphicsSettings.currentRenderPipeline as UniversalRenderPipelineAsset;
        
        var fld = typeof(UniversalRenderPipelineAsset)
                  .GetField("m_RendererDataList", BindingFlags.NonPublic | BindingFlags.Instance);
        
        var rendererDataList = fld.GetValue(urpAsset) as ScriptableRendererData[];
        if (rendererDataList == null || rendererDataList.Length == 0)
        {
            UnityEngine.Debug.LogError("No ScriptableRendererData found on URP asset.");
            return;
        }

        for (int i = 0; i < rendererDataList.Length; i++)
        {
            if (rendererDataList[i] == pvsRenderer)
            {
                rendererData = rendererDataList[i];
                rendererIndex = i;
                break;
            }
        }

        AddGVRenderFeature();
        AddPVVRenderFeature();

        _customDepthTexture = new RenderTexture(Screen.width, Screen.height, 32, RenderTextureFormat.Depth); // 24-bit depth buffer
        _customDepthTexture.name = "CustomGlobalDepth_RT";
        _customDepthTexture.filterMode = FilterMode.Point;
        _customDepthTexture.wrapMode = TextureWrapMode.Clamp;
        _customDepthTexture.Create();

        if (exportFrames && frameExportMode == ExportMode.Video)
            StartFFmpeg((mode == PVSMode.RenderPVV ? pvvColorFolder : gtColorFolder) + "/_rendering.mkv", Screen.width, Screen.height, target_fps);

        RenderPipelineManager.endCameraRendering += OnEndCameraRendering;
        RenderPipelineManager.beginContextRendering += OnBeginContextRendering;
        RenderPipelineManager.endContextRendering += OnEndContextRendering;
    }

    public void OnEndContextRendering(ScriptableRenderContext context, List<Camera> cameras)
    {
        if (Application.isPlaying && running && (targetSeconds > 0 && customFrameCount >= targetSeconds * target_fps))
        {
            // Check if we should stop the application
            bool shouldStop = mode == PVSMode.None || cur_radii_index >= radii.Count;
            
            // For RenderPVV mode, also check if there are no more PVV files to load
            if (mode == PVSMode.RenderPVV && !shouldStop)
            {
                string nextPvvFile = string.Format(pvvLoadFolder + "/{0}_predicted_pvv.bin.gz", ViewCell.InstanceCount);
                if (!System.IO.File.Exists(nextPvvFile))
                {
                    UnityEngine.Debug.Log($"No more PVV files found. Last attempted: {nextPvvFile}");
                    shouldStop = true;
                }
            }
            
            if (shouldStop)
            {
#if UNITY_EDITOR
                UnityEditor.EditorApplication.isPlaying = false;
#else
                    Application.Quit();
#endif
                running = false;
                cur_radii_index = 0;
            }
            else
            {
                OnDestroy();
                GameObject cinecam = GameObject.Find("CinemachineCamera");
                if (cinecam != null)
                    cinecam.GetComponent<CinemachineSplineDolly>().CameraPosition = 0.0f;
                else
                {
                    Animation animation = GetComponent<Animation>();
                    if (animation != null)
                    {
                        animation.Stop();
                        animation.Play();
                    }
                    else
                    {
                        Animator animator = GetComponent<Animator>();
                        if (animator != null)
                        animator.Play(animator.GetCurrentAnimatorStateInfo(0).shortNameHash, 0, 0f);
                    }
                }
                resetViewCell = true;
                ViewCell.InstanceCount = 0;
                Start();
            }
        }
    }

    public void ResetCounter()
    {
        if (discardCounterBuffer != null && discardCounterBuffer.IsValid())
        {
            _countData[0] = 0;
            discardCounterBuffer.SetData(_countData); // Send [0] back to GPU buffer
        }
    }

    /// <summary>
    /// Force ViewCell update and GV generation - useful when camera position changes significantly
    /// </summary>
    public void ForceViewCellUpdate()
    {
        resetViewCell = true;
        // The frame capture flag will be set in LateUpdate when resetViewCell is processed
    }

    public bool HasNeuralPVV
    {
        get { return neuralPVSHasPVV; }
    }

    public void ClearNeuralPVV()
    {
        neuralPVSHasPVV = false;
        neuralPVVViewCell = -1;
        neuralPVVTexture = null;
        Shader.SetGlobalTexture("_PVV", (Texture)null);
    }

    public void SetNeuralPVV(Texture3D pvvTexture, string sourcePath, int viewCellIndex)
    {
        neuralPVVTexture = pvvTexture;
        neuralPVVViewCell = viewCellIndex;
        neuralPVSHasPVV = true;
        Shader.SetGlobalTexture("_PVV", neuralPVVTexture);
        UnityEngine.Debug.Log($"Loaded NeuralPVS PVV for view cell {viewCellIndex}: {sourcePath}");
    }

    void AddGVRenderFeature()
    {
        if (gvFeature == null)
            gvFeature = ScriptableObject.CreateInstance<GeometryVolumeFeature>();     

        var existing = rendererData.rendererFeatures
                       .OfType<GeometryVolumeFeature>()
                       .FirstOrDefault();

        if (existing != null)
        {
            gvFeature = existing;
        }
        else
        {
            rendererData.rendererFeatures.Add(gvFeature);
        }

        gvFeature.volumeSettings = volumeSettings;
        gvFeature.gvSettings = geometryVolumeSettings;
        gvFeature.gvFolder = gvFolder;
        gvFeature.generateGV = (
            mode == PVSMode.GenerateGV ||
            mode == PVSMode.GenerateGVAndPVV ||
            mode == PVSMode.GenerateGVAndRenderNeuralPVV
        );
        gvFeature.updateGV = false;
        gvFeature.Create();

        rendererData.SetDirty();
    }

    void AddPVVRenderFeature()
    {
        if (pvvFeature == null)
            pvvFeature = ScriptableObject.CreateInstance<PotentiallyVisibleVolumeFeature>();

        var existing = rendererData.rendererFeatures
                       .OfType<PotentiallyVisibleVolumeFeature>()
                       .FirstOrDefault();

        if (existing != null)
        {
            pvvFeature = existing;
        }
        else
        {
            rendererData.rendererFeatures.Add(pvvFeature);
        }

        pvvFeature.volumeSettings = volumeSettings;
        pvvFeature.pvvSettings = potentiallyVisibleVolumeSettings;
        pvvFeature.pvvFolder = pvvFolder;
        pvvFeature.generatePVV = (mode == PVSMode.GeneratePVV || mode == PVSMode.GenerateGVAndPVV);
        pvvFeature.updatePVV = false;
        pvvFeature.Create();

        rendererData.SetDirty();
    }

    private void OnDisable()
    {
        if (discardCounterBuffer != null)
        {
            discardCounterBuffer.Release();
            discardCounterBuffer = null;
        }
        Shader.SetGlobalBuffer("_DiscardCounter", (ComputeBuffer)null);
    }

    private void OnDestroy()
    {
        RenderPipelineManager.endCameraRendering -= OnEndCameraRendering;
        RenderPipelineManager.beginContextRendering -= OnBeginContextRendering;

        Shader.SetGlobalInt("_CheckPVV", 0);

        rendererData.rendererFeatures.Remove(gvFeature);
        rendererData.rendererFeatures.Remove(pvvFeature);
        rendererData.SetDirty();

        if (exportFrames && frameExportMode == ExportMode.Video)
            StopFFmpeg();

        UnityEngine.Debug.Log($"Avg. GV timing: {gvTimings/gvSamples}");
        UnityEngine.Debug.Log($"GV samples: {gvSamples}");

        UnityEngine.Debug.Log($"Avg. GPU timing: {frameTimings / timingCounter}");
        UnityEngine.Debug.Log($"GPU samples: {timingCounter}");
    }

    private void Update()
    {
        FrameTimingManager.CaptureFrameTimings();

        if (GeometryVolumeFeature.k_GenerateGeomSampler.gpuSampleCount > 0)
        {
            UnityEngine.Debug.Log($"[GPU] GenerateGeometryVolume SS: {GeometryVolumeFeature.k_GenerateGeomSampler.gpuElapsedTime:F3} ms over {GeometryVolumeFeature.k_GenerateGeomSampler.gpuSampleCount} calls");
            gvTimings += GeometryVolumeFeature.k_GenerateGeomSampler.gpuElapsedTime;
            gvSamples += GeometryVolumeFeature.k_GenerateGeomSampler.gpuSampleCount;
        }

        var ret = FrameTimingManager.GetLatestTimings((uint)m_FrameTimings.Length, m_FrameTimings);
        if (ret > 0 && m_FrameTimings[0].gpuFrameTime > 0.0 && m_FrameTimings[0].gpuFrameTime < 1.0)
        {
            frameTimings += (float)m_FrameTimings[0].gpuFrameTime;
            timingCounter++;
        }
    }

    void LateUpdate()
    {   
        customFrameCount++;

        if (Application.isPlaying && mode != PVSMode.None)
        {
            bool viewcelltest = volumeSettings.viewCell == null;
            bool incelltest = !viewcelltest ? volumeSettings.viewCell.IsInCell(GetComponent<Camera>().transform.position, 0.01f) : true;
            bool rotationtest = !viewcelltest ? Quaternion.Angle(GetComponent<Camera>().transform.rotation, volumeSettings.viewCell.GetCenterCam().transform.rotation) > 15f : false;

            bool updateViewCell = (volumeSettings.viewCell == null || !volumeSettings.viewCell.IsInCell(GetComponent<Camera>().transform.position, 0.01f)
                || Quaternion.Angle(GetComponent<Camera>().transform.rotation, volumeSettings.viewCell.GetCenterCam().transform.rotation) > 15f);

            
            if (updateViewCell || resetViewCell)
            {
                if (resetViewCell)
                {
                    resetViewCell = false;
                    // If this is a manual reset and we're exporting frames in GV mode, ensure capture
                    if ((mode == PVSMode.GenerateGV || mode == PVSMode.GenerateGVAndPVV) && exportFrames && !shouldCaptureFrameThisFrame)
                    {
                        shouldCaptureFrameThisFrame = true;
                        UnityEngine.Debug.Log("Manual ViewCell reset detected, frame capture enabled");
                    }
                }

                if (volumeSettings.viewCell == null)
                    volumeSettings.viewCell = new ViewCell(viewCellSettings, GetComponent<Camera>(), mode, rendererIndex);
                else
                    volumeSettings.viewCell.UpdateViewCell(GetComponent<Camera>());

                volumeSettings.viewCell.EnableSampleCams();

                if (mode == PVSMode.GenerateGV || mode == PVSMode.GenerateGVAndRenderNeuralPVV)
                {
                    if (mode == PVSMode.GenerateGVAndRenderNeuralPVV)
                        ClearNeuralPVV();

                    gvFeature.updateGV = true;
                    // Set flag to capture frame during this GV generation
                    if (exportFrames && !shouldCaptureFrameThisFrame)
                    {
                        shouldCaptureFrameThisFrame = true;
                        UnityEngine.Debug.Log($"GV generation started, frame capture enabled for ViewCell #{ViewCell.InstanceCount}");
                    }
                }
                else if (mode == PVSMode.GeneratePVV)
                {
                    pvvFeature.updatePVV = true;
                }
                else if (mode == PVSMode.GenerateGVAndPVV)
                {
                    gvFeature.updateGV = true;
                    pvvFeature.updatePVV = true;
                    // Set flag to capture frame during this GV and PVV generation
                    if (exportFrames && !shouldCaptureFrameThisFrame)
                    {
                        shouldCaptureFrameThisFrame = true;
                        UnityEngine.Debug.Log($"GV and PVV generation started, frame capture enabled for ViewCell #{ViewCell.InstanceCount}");
                    }
                }
                else if (mode == PVSMode.RenderPVV && ViewCell.InstanceCount > 0)
                {
                    string pvvFilePath = string.Format(pvvLoadFolder + "/{0}_predicted_pvv.bin.gz", ViewCell.InstanceCount - 1);
                    if (System.IO.File.Exists(pvvFilePath))
                    {
                        Texture3D pvvTexture = Utils.LoadPVV(pvvFilePath, volumeSettings.volumeSize, volumeSettings.volumeSize, volumeSettings.volumeDepth);
                        if (pvvTexture != null)
                        {
                            Shader.SetGlobalTexture("_PVV", pvvTexture);
                            UnityEngine.Debug.Log($"Loaded PVV file: {pvvFilePath}");
                        }
                        else
                        {
                            UnityEngine.Debug.LogWarning($"Failed to load PVV texture from: {pvvFilePath}");
                        }
                    }
                    else
                    {
                        UnityEngine.Debug.LogWarning($"PVV file not found: {pvvFilePath}");
                    }

                    // Uncomment to save imported pvv (check for correct loading)
                    //NativeArray<byte> pvvData = pvvTexture.GetPixelData<byte>(0);
                    //_ = Task.Run(async() => await Utils.WriteGzipStreamedAsync(string.Format(pvvLoadFolder + "/{0}_test_pvv.bin.gz", ViewCell.InstanceCount - 1), pvvData));
                }
            }
            else
            {
                if (mode != PVSMode.RenderPVV && mode != PVSMode.GenerateGVAndRenderNeuralPVV && volumeSettings.viewCell != null)
                    volumeSettings.viewCell.DisableSampleCams();
            }
        }
    }

    void OnBeginContextRendering(ScriptableRenderContext context, List<Camera> cameras)
    {
        if (Application.isPlaying && mode != PVSMode.None && Camera.main.cameraType == CameraType.Game && volumeSettings.viewCell != null)
        {
            float pvvFarPlane = viewCellSettings.pvvFarPlane;
            float vcFarPlane = volumeSettings.viewCell.GetFarPlane();
            bool useLoadedPVV = mode == PVSMode.RenderPVV && ViewCell.InstanceCount > 0;
            bool useNeuralPVV = mode == PVSMode.GenerateGVAndRenderNeuralPVV && neuralPVSHasPVV;
            
            Shader.SetGlobalFloat("_PVVFarPlane", pvvFarPlane);
            Shader.SetGlobalInt("_SamplingFactor", volumeSettings.samplingFactor);
            Shader.SetGlobalInt("_LinearZ", volumeSettings.linearZ ? 1 : 0);
            Shader.SetGlobalInt("_CheckPVV", useLoadedPVV || useNeuralPVV ? (int)filter : 0);
            Shader.SetGlobalInt("_PVVMode", (int)pvsRenderMode);
            Shader.SetGlobalMatrix("_ViewCellV", volumeSettings.viewCell.GetViewMatrix());
            Shader.SetGlobalMatrix("_ViewCellP", volumeSettings.viewCell.GetProjectionMatrix(volumeSettings.linearZ));
            Shader.SetGlobalFloat("_VCNearPlane", volumeSettings.viewCell.GetNearPlane());
            Shader.SetGlobalFloat("_VCFarPlane", vcFarPlane);
            Shader.SetGlobalFloat("_LogScale", volumeSettings.logDepthScale);
            Shader.SetGlobalMatrix("_MainCamVP", GL.GetGPUProjectionMatrix(Camera.main.projectionMatrix, true) * Camera.main.worldToCameraMatrix);
        }
        else
            Shader.SetGlobalInt("_CheckPVV", 0);
    }

    void OnEndCameraRendering(ScriptableRenderContext ctx, Camera cam)
    {
        bool correct_camera = false;

        if (mode != PVSMode.RenderPVV && cam.tag == "MainCamera")
            correct_camera = true;
        else if (mode == PVSMode.RenderPVV && ((pvsRenderMode == PVVRenderMode.FirstPerson && cam.tag == "MainCamera") || (pvsRenderMode == PVVRenderMode.ThirdPerson && cam.name.Contains("ThirdPerson"))))
            correct_camera = true;

        // For GV generation mode, only export frames when GV is being generated
        bool shouldExportFrame = exportFrames && Application.isPlaying && correct_camera;
        if ((mode == PVSMode.GenerateGV || mode == PVSMode.GenerateGVAndPVV || mode == PVSMode.GenerateGVAndRenderNeuralPVV) && shouldExportFrame)
        {
            // Only export frame if we're flagged to capture this frame (when GV generation is active)
            shouldExportFrame = shouldCaptureFrameThisFrame;
        }

        if (shouldExportFrame)
        {
            uint[] result = new uint[1];
            discardCounterBuffer.GetData(result);

            discardCounterBuffer.SetData(new uint[1] { 0 });

            var rt = RenderTexture.GetTemporary(Screen.width, Screen.height, 0, GraphicsFormat.R8G8B8A8_UNorm);
            ScreenCapture.CaptureScreenshotIntoRenderTexture(rt);
            if (frameExportMode == ExportMode.Video)
            {
                Utils.SaveRenderTextureToVideo(ffmpegIn, rt, 4, false);
                UnityEngine.Debug.Log($"Frame {file_count:D4} saved to video stream (GV #{ViewCell.InstanceCount}): {(mode == PVSMode.RenderPVV || mode == PVSMode.GenerateGVAndRenderNeuralPVV ? pvvColorFolder : gtColorFolder)}/_rendering.mkv");
            }
            else
            {
                var rt_flipped = RenderTexture.GetTemporary(Screen.width, Screen.height, 0, GraphicsFormat.R8G8B8A8_UNorm);
                Graphics.Blit(rt, rt_flipped, new Vector2(1.0f, -1.0f), new Vector2(0.0f, 1.0f));
                string filePath = ((mode == PVSMode.GenerateGV || mode == PVSMode.GenerateGVAndPVV) ? gtColorFolder : pvvColorFolder) + string.Format("/{0}.png", file_count.ToString("D4"));
                Utils.SaveRenderTextureToFile(rt_flipped, filePath, 4, false);
                UnityEngine.Debug.Log($"Frame {file_count:D4} saved (GV #{ViewCell.InstanceCount}): {filePath}");
                RenderTexture.ReleaseTemporary(rt_flipped);
            }
            RenderTexture.ReleaseTemporary(rt);
            file_count++;
            
            // Reset the flag after capturing
            shouldCaptureFrameThisFrame = false;
        }

        if (Application.isPlaying && customFrameCount == 10 && cam.name.Contains("ScreenshotCam"))
        {
            UnityEngine.Debug.Log("Saving Scene Screenshot");
            var rt = RenderTexture.GetTemporary(Screen.width, Screen.height, 0, GraphicsFormat.R8G8B8A8_UNorm);
            var rt_flipped = RenderTexture.GetTemporary(Screen.width, Screen.height, 0, GraphicsFormat.R8G8B8A8_UNorm);
            ScreenCapture.CaptureScreenshotIntoRenderTexture(rt);
            Graphics.Blit(rt, rt_flipped, new Vector2(1.0f, -1.0f), new Vector2(0.0f, 1.0f));
            string screenshotPath = gtColorFolder + $"/scene_{scene}.png";
            Utils.SaveRenderTextureToFile(rt_flipped, screenshotPath, 4, false);
            UnityEngine.Debug.Log($"Scene screenshot saved to: {screenshotPath}");
            RenderTexture.ReleaseTemporary(rt);
            RenderTexture.ReleaseTemporary(rt_flipped);
        }

        if (Application.isPlaying && (mode == PVSMode.RenderPVV || mode == PVSMode.GenerateGVAndRenderNeuralPVV) && pvsRenderMode == PVVRenderMode.ThirdPerson && cam.tag == "MainCamera")
        {
            Graphics.Blit(Shader.GetGlobalTexture("_CameraDepthTexture"), _customDepthTexture);
            Shader.SetGlobalTexture(globalTextureName, _customDepthTexture);
            Shader.SetGlobalMatrix("_MainCamVP", GL.GetGPUProjectionMatrix(cam.projectionMatrix, true) * cam.worldToCameraMatrix);
        }
    }

    private string FindFFmpeg()
    {
        string[] paths = Environment.GetEnvironmentVariable("PATH").Split(Path.PathSeparator);
        foreach (string path in paths)
        {
            string ffmpegPath = Path.Combine(path, "ffmpeg.exe");
            if (File.Exists(ffmpegPath))
                return ffmpegPath;
        }
        // Fallback to common locations
        string[] commonPaths = {
            @"C:\ffmpeg\bin\ffmpeg.exe",
            @"C:\Program Files\ffmpeg\bin\ffmpeg.exe"
        };
        foreach (string path in commonPaths)
        {
            if (File.Exists(path))
                return path;
        }
        // Check WinGet location
        string wingetPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), @"Microsoft\WinGet\Packages");
        if (Directory.Exists(wingetPath))
        {
            var dirs = Directory.GetDirectories(wingetPath, "Gyan.FFmpeg*");
            if (dirs.Length > 0)
            {
                string binDir = Path.Combine(dirs[0], "ffmpeg-*-full_build", "bin");
                if (Directory.Exists(binDir.Replace("*-full_build", "").Replace("*", ""))) // rough check
                {
                    var files = Directory.GetFiles(binDir.Replace("*-full_build", "").Replace("*", ""), "ffmpeg.exe", SearchOption.AllDirectories);
                    if (files.Length > 0)
                        return files[0];
                }
                else
                {
                    var subDirs = Directory.GetDirectories(dirs[0], "ffmpeg-*-full_build");
                    if (subDirs.Length > 0)
                    {
                        string binPath = Path.Combine(subDirs[0], "bin", "ffmpeg.exe");
                        if (File.Exists(binPath))
                            return binPath;
                    }
                }
            }
        }
        return null;
    }

    private void StartFFmpeg(string path, int width, int height, int fps)
    {
        string ffmpegPath = FindFFmpeg();
        if (string.IsNullOrEmpty(ffmpegPath))
        {
            UnityEngine.Debug.LogError("FFmpeg executable not found. Please install FFmpeg and ensure it's in your PATH.");
            return;
        }

        string args = $"-hide_banner -loglevel error -y -framerate {target_fps} -f rawvideo -vcodec rawvideo -pixel_format rgba -video_size {width}x{height} " +
                      $"-i - " +
                      $"-vsync cfr -pixel_format rgba -c:v hevc_nvenc -tune lossless -rc constqp -pix_fmt gbrp -bsf:v \"hevc_metadata=video_full_range_flag=1\" " +
                      $"-an \"{path}\"";

        var utf8NoBom = new System.Text.UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
        
        var psi = new ProcessStartInfo
        {
            FileName = ffmpegPath,
            Arguments = args,
            RedirectStandardInput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            StandardInputEncoding = utf8NoBom,
        };

        ffmpeg = new Process { StartInfo = psi };
        ffmpeg.Start();
        ffmpegIn = ffmpeg.StandardInput.BaseStream;

        ffmpeg.ErrorDataReceived += (sender, e) =>
        {
            if (!string.IsNullOrEmpty(e.Data))
                UnityEngine.Debug.LogError("[FFmpeg] " + e.Data);
        };

        ffmpeg.BeginErrorReadLine();
    }

    private void StopFFmpeg()
    {
        try
        {
            AsyncGPUReadback.WaitAllRequests();

            ffmpegIn.Flush();
            ffmpegIn.Close();

            ffmpeg.WaitForExit();
            ffmpeg.Close();
            ffmpeg.Dispose();
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogError("Error closing FFmpeg: " + e);
        }
    }
}
