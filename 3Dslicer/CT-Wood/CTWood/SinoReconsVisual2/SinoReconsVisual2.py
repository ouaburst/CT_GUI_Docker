import os
import json
import logging
import requests
import qt
import slicer
import vtk

from urllib.parse import urlparse
from qt import QProgressDialog
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from qt import QTimer

cached_full_trajectory = None

# -----------------------------
# Settings helpers
# -----------------------------
SETTINGS_KEY_BACKEND_URL = "SinoReconsVisual2/BackendUrl"
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"

def get_saved_base_url() -> str:
    """Return the persisted backend base URL or default."""
    return qt.QSettings().value(SETTINGS_KEY_BACKEND_URL, DEFAULT_BACKEND_URL)

def set_saved_base_url(url: str) -> None:
    """Persist the backend base URL."""
    qt.QSettings().setValue(SETTINGS_KEY_BACKEND_URL, url)

def normalize_base_url(url: str) -> str:
    """Trim spaces and trailing slashes; add http:// if missing scheme."""
    url = (url or "").strip()
    if not url:
        return DEFAULT_BACKEND_URL
    if "://" not in url:
        url = "http://" + url
    return url.rstrip("/")

def _normalize_fbp_filter(name: str) -> str:
    """Map UI filter labels to ODL-accepted names."""
    key = (name or "").strip().lower()
    mapping = {
        "ram-lak": "ram-lak",
        "ram lak": "ram-lak",
        "ramlak": "ram-lak",
        "shepp-logan": "shepp-logan",
        "shepp logan": "shepp-logan",
        "cosine": "cosine",
        "hamming": "hamming",
        "hann": "hann",
    }
    return mapping.get(key, "ram-lak")

class SinoReconsVisual2(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("SinoReconsVisual2")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Utilities")]
        self.parent.dependencies = []
        self.parent.contributors = ["Oualid Burström (LTU)"]
        self.parent.helpText = _("The SinoReconsVisual2 extension enables users ...")
        self.parent.acknowledgementText = _("")
        iconsPath = os.path.join(os.path.dirname(__file__), "Resources", "Icons")
        iconPath = os.path.join(iconsPath, "SinoReconsVisual2.png")
        if os.path.exists(iconPath):
            self.parent.icon = qt.QIcon(iconPath)


class SinoReconsVisual2Widget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)

    # -----------------------------
    # UI & wiring
    # -----------------------------
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # Load and attach UI
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SinoReconsVisual2.ui"))
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        self.layout.addWidget(uiWidget)

        # Initialize saved backend URL into the line edit (new field in UI)
        if hasattr(self.ui, "serverUrlLineEdit") and self.ui.serverUrlLineEdit is not None:
            self.ui.serverUrlLineEdit.setText(get_saved_base_url())

        # Initialize window size
        self.windowSize = 1
        self.currentIndex = 0

        # Connect signals to slots
        self.ui.indexSlider.valueChanged.connect(self.onIndexChanged)
        self.ui.executeRemoteCommandButton.clicked.connect(self.onExecuteRemoteCommandClicked)
        self.ui.streamNrrdButton.clicked.connect(self.onStreamNrrdButtonClicked)
        self.ui.connectToServerButton.clicked.connect(self.onConnectToServerClicked)
        self.ui.reconstructionSelectorComboBox.currentTextChanged.connect(self.onReconstructionMethodChanged)
        self.ui.loadSourceTrajectoryButton.clicked.connect(self.onLoadSourceTrajectoryClicked)
        self.ui.loadFullDatasetButton.clicked.connect(self.onLoadFullDataset)
        self.ui.indexSlider.valueChanged.connect(self.onSliderChanged)
        self.ui.fetchSinogramButton.clicked.connect(self.onFetchSinogramSliceClicked)

        # Slider debounce timer
        self.sliderDebounceTimer = QTimer()
        self.sliderDebounceTimer.setSingleShot(True)
        self.sliderDebounceTimer.setInterval(300)  # milliseconds
        self.sliderDebounceTimer.timeout.connect(self.onFetchSinogramSliceClicked)

    # -----------------------------
    # Utility
    # -----------------------------
    def _currentBaseUrl(self) -> str:
        """Read base URL from line edit (if present) or settings; normalize and persist."""
        if hasattr(self.ui, "serverUrlLineEdit") and self.ui.serverUrlLineEdit is not None:
            entered = self.ui.serverUrlLineEdit.text
            base = normalize_base_url(entered)
            # Persist any change immediately so other actions use it
            set_saved_base_url(base)
            # Keep UI clean/normalized
            if entered != base and self.ui.serverUrlLineEdit.text != base:
                self.ui.serverUrlLineEdit.setText(base)
            return base
        # Fallback to settings
        return normalize_base_url(get_saved_base_url())

    def clearModels(self):
        modelNamesToKeep = {"SourceTrajectory"}
        scene = slicer.mrmlScene
        for node in scene.GetNodesByClass("vtkMRMLModelNode"):
            if node.GetName() not in modelNamesToKeep:
                scene.RemoveNode(node)

    # -----------------------------
    # UI reactions
    # -----------------------------
    def onReconstructionMethodChanged(self, method):
        method = (method or "").lower()
        self.ui.fbpParameterGroupBox.setVisible(method == "fbp")
        self.ui.landweberParameterGroupBox.setVisible(method == "landweber")

    def _parse_selected_sample(self):
        """Return (tree_ID, disk_ID) from UI selection."""
        sample_text = self.ui.sampleSelectorComboBox.currentText
        parts = sample_text.replace("Tree", "").replace("Disk", "").split("-")
        tree_ID = int(parts[0].strip())
        disk_ID = int(parts[1].strip())
        return tree_ID, disk_ID

    def onStreamNrrdButtonClicked(self):
        try:
            tree_ID, disk_ID = self._parse_selected_sample()
            method = (self.ui.reconstructionSelectorComboBox.currentText or "").lower()
            filename = f"tree{tree_ID}_disk{disk_ID}_{method}.nrrd"
            base = self._currentBaseUrl()
            stream_nrrd_from_url(filename, base)
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to parse sample.\nError: {e}", windowTitle="Parsing Error")

    def onExecuteRemoteCommandClicked(self):
        """
        Call backend /run_reconstruction and auto-load the produced NRRD.
        Only send progress_* parameters for iterative methods (e.g., landweber).
        """
        base = self._currentBaseUrl()
        run_url = f"{base}/run_reconstruction"

        # specie is fixed for now; make it a UI control later if needed
        specie = "pine"

        # Parse sample selection: "Tree X - Disk Y"
        try:
            tree_ID, disk_ID = self._parse_selected_sample()
        except Exception:
            slicer.util.errorDisplay(
                "Failed to parse selected sample (Tree/Disk).",
                windowTitle="Parsing Error"
            )
            return

        # Method + parameters
        method = (self.ui.reconstructionSelectorComboBox.currentText or "").lower()
        params = {}

        if method == "fbp":
            # Map UI filter to ODL name
            try:
                filt = _normalize_fbp_filter(self.ui.filterTypeComboBox.currentText)
            except NameError:
                # fallback if helper not present
                filt = (self.ui.filterTypeComboBox.currentText or "ram-lak").strip().lower().replace(" ", "-")
            params["filter_type"] = filt
            params["frequency_scaling"] = float(self.ui.frequencyScalingSpinBox.value)
            if self.ui.paddingCheckBox.isChecked():
                params["padding"] = True

            # IMPORTANT: do NOT add progress_* to FBP
        elif method == "landweber":
            params["niter"] = int(self.ui.iterationsSpinBox.value)
            params["omega"] = float(self.ui.relaxationSpinBox.value)

            # progress_* only for iterative methods
            try:
                progress_every = int(self.ui.progressEverySpinBox.value)
            except Exception:
                progress_every = 5
            params["progress_every"] = progress_every
            params["progress_path"] = os.path.join(slicer.app.temporaryPath, "recon_progress.jsonl")
        else:
            # adjoint or other single-shot methods: no extra params
            pass

        payload = {
            "specie": specie,
            "tree_ID": int(tree_ID),
            "disk_ID": int(disk_ID),
            "method": method,
            "parameters": params,
        }

        # Simple progress UI (parses "%” from streamed lines if present)
        progressDialog = qt.QProgressDialog("Running reconstruction...", "Cancel", 0, 100)
        progressDialog.setWindowTitle("Reconstruction progress")
        progressDialog.setMinimumDuration(0)
        progressDialog.setValue(0)
        canceled = [False]
        progressDialog.canceled.connect(lambda: canceled.__setitem__(0, True))

        try:
            import requests, re, json as _json
            slicer.util.infoDisplay("Starting remote reconstruction…", "Reconstruction")

            r = requests.post(run_url, json=payload, stream=True, timeout=None)
            r.raise_for_status()

            pct_re = re.compile(r"\((\d+(?:\.\d+)?)%\)")
            output_url = None

            # If server returns application/json (non-stream), pretty print it once
            ctype = r.headers.get("Content-Type", "")
            if "application/json" in ctype and not r.headers.get("Transfer-Encoding") == "chunked":
                try:
                    resp = r.json()
                    print(_json.dumps(resp, indent=2))
                    if isinstance(resp, dict) and "url" in resp:
                        output_url = resp["url"]
                except Exception:
                    print(r.text)
            else:
                # Streamed logs
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    print(line)

                    # Update progress if we see "(12.5%)" pattern (iterative methods)
                    m = pct_re.search(line)
                    if m:
                        try:
                            progressDialog.setValue(int(float(m.group(1))))
                            slicer.app.processEvents()
                            if canceled[0]:
                                slicer.util.errorDisplay("Reconstruction canceled.", windowTitle="Reconstruction")
                                return
                        except Exception:
                            pass

                    # Capture explicit output hint: "Output: /images/xxx.nrrd"
                    if "Output:" in line:
                        output_url = line.split("Output:", 1)[1].strip()

            progressDialog.setValue(100)

            # Determine filename to load from /images
            if output_url:
                filename = output_url.rsplit("/", 1)[-1]
            else:
                filename = f"tree{tree_ID}_disk{disk_ID}_{method}.nrrd"

            stream_nrrd_from_url(filename, base)

        except requests.exceptions.RequestException as e:
            slicer.util.errorDisplay(
                f"Failed to start reconstruction at:\n{run_url}\n\n{e}",
                windowTitle="Reconstruction Error"
            )
        except Exception as e:
            slicer.util.errorDisplay(f"Reconstruction failed:\n{e}", windowTitle="Reconstruction Error")
        

    def onConnectToServerClicked(self):
        # Persist (and normalize) what the user entered before we request
        base = self._currentBaseUrl()
        config_url = f"{base}/slicer_backend_config.json"

        try:
            response = requests.get(config_url, timeout=5)
            response.raise_for_status()
            config = response.json()

            # Populate container/volume for completeness (from config)
            container_name = config.get("container_name", "")
            self.ui.containerSelectorComboBox.clear()
            if container_name:
                self.ui.containerSelectorComboBox.addItem(container_name)

            volume_name = config.get("volume_name", "")
            self.ui.volumeSelectorComboBox.clear()
            if volume_name:
                self.ui.volumeSelectorComboBox.addItem(volume_name)

            # Populate sample combo box
            samples = config.get("samples", [])
            self.ui.sampleSelectorComboBox.clear()
            for sample in samples:
                # expect keys: specie, tree_ID, disk_ID
                sample_text = f"Tree {sample['tree_ID']} - Disk {sample['disk_ID']}"
                self.ui.sampleSelectorComboBox.addItem(sample_text)

            # Populate reconstruction methods
            recons = config.get("reconstruction_methods", [])
            self.ui.reconstructionSelectorComboBox.clear()
            for r in recons:
                self.ui.reconstructionSelectorComboBox.addItem(r)

            # Initial visibility for parameter groups
            selected_method = self.ui.reconstructionSelectorComboBox.currentText
            self.onReconstructionMethodChanged(selected_method)

            # FBP params
            fbp_params = config.get("fbp_parameters", {})
            self.ui.filterTypeComboBox.clear()
            for f in fbp_params.get("filter_type", []):
                self.ui.filterTypeComboBox.addItem(f)

            freq = fbp_params.get("frequency_scaling", {})
            self.ui.frequencyScalingSpinBox.setMinimum(freq.get("min", 0.0))
            self.ui.frequencyScalingSpinBox.setMaximum(freq.get("max", 1.0))
            self.ui.frequencyScalingSpinBox.setSingleStep(freq.get("step", 0.05))
            self.ui.frequencyScalingSpinBox.setValue(freq.get("default", 1.0))
            self.ui.paddingCheckBox.setChecked("Yes" in fbp_params.get("padding", []))

            # Landweber params
            landweber_params = config.get("landweber_parameters", {})
            iters = landweber_params.get("iterations", {})
            self.ui.iterationsSpinBox.setMinimum(iters.get("min", 1))
            self.ui.iterationsSpinBox.setMaximum(iters.get("max", 500))
            self.ui.iterationsSpinBox.setSingleStep(iters.get("step", 10))
            self.ui.iterationsSpinBox.setValue(iters.get("default", 100))

            relax = landweber_params.get("relaxation", {})
            self.ui.relaxationSpinBox.setMinimum(relax.get("min", 0.001))
            self.ui.relaxationSpinBox.setMaximum(relax.get("max", 1.0))
            self.ui.relaxationSpinBox.setSingleStep(relax.get("step", 0.01))
            self.ui.relaxationSpinBox.setValue(relax.get("default", 0.2))

            self.ui.statusLabel.setText("Status: Connected")

        except requests.exceptions.RequestException as e:
            self.ui.statusLabel.setText("Status: Failed to connect")
            slicer.util.errorDisplay(f"Could not fetch config from:\n{config_url}\n\n{e}", windowTitle="Connection Error")

    def onLoadSourceTrajectoryClicked(self):
        try:
            self.clearModels()
            base = self._currentBaseUrl()
            # If you later export VTPs to /files/, list them here:
            stream_vtp_from_url([
                "fov_rays.vtp",
                "source_path.vtp",
                "detector_path.vtp"
            ], base)
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to load geometry: {e}")

    def onSliderChanged(self, value):
        self.ui.sliderIndexLabel.setText(f"Index: {value}")
        self.updateViewFromFullDataset(value)

    def printCameraDebugInfo(self, cameraNode, label=""):
        pos = cameraNode.GetPosition()
        fp = cameraNode.GetFocalPoint()
        view_up = cameraNode.GetViewUp()
        view_angle = cameraNode.GetViewAngle()
        distance = ((pos[0]-fp[0])**2 + (pos[1]-fp[1])**2 + (pos[2]-fp[2])**2) ** 0.5

        print(f"\n[DEBUG] Camera Info {label}")
        print(f"Position     : {pos}")
        print(f"Focal Point  : {fp}")
        print(f"View Up      : {view_up}")
        print(f"View Angle   : {view_angle}")
        print(f"Distance     : {distance:.3f}\n")

    # -----------------------------
    # Rendering helpers
    # -----------------------------
    def createPointsModel(self, points, name, color=(1,1,1), pointSize=3):
        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        points_vtk = vtk.vtkPoints()
        for p in points:
            points_vtk.InsertNextPoint(p)
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points_vtk)
        verts = vtk.vtkCellArray()
        for i in range(len(points)):
            verts.InsertNextCell(1)
            verts.InsertCellPoint(i)
        polydata.SetVerts(verts)
        modelNode.SetAndObservePolyData(polydata)

        displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        slicer.mrmlScene.AddNode(displayNode)
        modelNode.SetAndObserveDisplayNodeID(displayNode.GetID())
        displayNode.SetColor(color)
        displayNode.SetPointSize(pointSize)
        displayNode.SetVisibility(1)

    def createQuadsModel(self, quads, name, color=(1,1,1), lineWidth=2):
        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        points_vtk = vtk.vtkPoints()
        polys = vtk.vtkCellArray()
        pointIndex = 0
        for quad in quads:
            for p in quad:
                points_vtk.InsertNextPoint(p)
            cell = vtk.vtkQuad()
            for i in range(4):
                cell.GetPointIds().SetId(i, pointIndex + i)
            polys.InsertNextCell(cell)
            pointIndex += 4
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points_vtk)
        polydata.SetPolys(polys)
        modelNode.SetAndObservePolyData(polydata)

        displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        slicer.mrmlScene.AddNode(displayNode)
        modelNode.SetAndObserveDisplayNodeID(displayNode.GetID())
        displayNode.SetColor(color)
        displayNode.SetEdgeVisibility(1)
        displayNode.SetLineWidth(lineWidth)
        displayNode.SetVisibility(1)

    def createLinesModel(self, lines, name, color=(1,1,1), lineWidth=2):
        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        points_vtk = vtk.vtkPoints()
        lines_vtk = vtk.vtkCellArray()
        idx = 0
        for p1, p2 in lines:
            points_vtk.InsertNextPoint(p1)
            points_vtk.InsertNextPoint(p2)
            lines_vtk.InsertNextCell(2)
            lines_vtk.InsertCellPoint(idx)
            lines_vtk.InsertCellPoint(idx + 1)
            idx += 2
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points_vtk)
        polydata.SetLines(lines_vtk)
        modelNode.SetAndObservePolyData(polydata)

        displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        slicer.mrmlScene.AddNode(displayNode)
        modelNode.SetAndObserveDisplayNodeID(displayNode.GetID())
        displayNode.SetColor(color)
        displayNode.SetLineWidth(lineWidth)
        displayNode.SetVisibility(1)

    def createTrajectoryLine(self, points, name="SourceTrajectory", color=(0.0, 0.4, 1.0), lineWidth=3):
        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        points_vtk = vtk.vtkPoints()
        lines = vtk.vtkCellArray()

        for p in points:
            points_vtk.InsertNextPoint(p)

        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(points))
        for i in range(len(points)):
            line.GetPointIds().SetId(i, i)
        lines.InsertNextCell(line)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points_vtk)
        polydata.SetLines(lines)
        modelNode.SetAndObservePolyData(polydata)

        displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        slicer.mrmlScene.AddNode(displayNode)
        modelNode.SetAndObserveDisplayNodeID(displayNode.GetID())
        displayNode.SetColor(color)
        displayNode.SetLineWidth(lineWidth)
        displayNode.SetVisibility(1)

    def createBezierSurfaceModel(self, curves, name="DetectorSurface", color=(1.0, 0.5, 0.0), opacity=0.7):
        if not curves:
            print("[WARNING] Empty curve list passed to createBezierSurfaceModel.")
            return

        num_rows = len(curves)
        num_cols = len(curves[0])
        if any(len(row) != num_cols for row in curves):
            print("[ERROR] Inconsistent number of points per curve row.")
            return

        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        points_vtk = vtk.vtkPoints()
        polys = vtk.vtkCellArray()

        for row in curves:
            for pt in row:
                points_vtk.InsertNextPoint(pt)

        def idx(i, j):
            return i * num_cols + j

        for i in range(num_rows - 1):
            for j in range(num_cols - 1):
                quad = vtk.vtkQuad()
                quad.GetPointIds().SetId(0, idx(i, j))
                quad.GetPointIds().SetId(1, idx(i, j + 1))
                quad.GetPointIds().SetId(2, idx(i + 1, j + 1))
                quad.GetPointIds().SetId(3, idx(i + 1, j))
                polys.InsertNextCell(quad)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points_vtk)
        polydata.SetPolys(polys)
        modelNode.SetAndObservePolyData(polydata)

        displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        slicer.mrmlScene.AddNode(displayNode)
        modelNode.SetAndObserveDisplayNodeID(displayNode.GetID())

        displayNode.SetColor(color)
        displayNode.SetOpacity(opacity)
        displayNode.SetVisibility(1)
        displayNode.SetRepresentation(2)          # Surface
        displayNode.SetEdgeVisibility(False)      # Hide edges
        displayNode.SetInterpolation(1)           # Phong
        displayNode.SetBackfaceCulling(False)     # Render both sides

    # -----------------------------
    # Full dataset workflow
    # -----------------------------
    def onLoadFullDataset(self):
        try:
            self.clearModels()
            base = self._currentBaseUrl()
            url = f"{base}/full_geometry"
            response = requests.get(url, timeout=300)  # Up to 5 minutes
            response.raise_for_status()
            self.fullGeometryData = response.json()

            total_frames = len(self.fullGeometryData.get("sources", []))

            print(f"[INFO] Loaded full geometry: "
                  f"{total_frames} sources, "
                  f"{len(self.fullGeometryData.get('detector_panels', []))} panels")

            # Configure slider
            self.ui.indexSlider.setMinimum(0)
            self.ui.indexSlider.setMaximum(max(0, total_frames - 1))
            self.ui.indexSlider.setValue(0)
            self.ui.sliderIndexLabel.setText("Index: 0")

            # Load first frame
            self.updateViewFromFullDataset(0)

        except Exception as e:
            slicer.util.errorDisplay(f"Failed to load full dataset:\n{e}")

    def updateViewFromFullDataset(self, index):
        if not hasattr(self, "fullGeometryData"):
            slicer.util.errorDisplay("Geometry data not loaded. Call onLoadFullDataset first.")
            return

        try:
            sources = self.fullGeometryData.get("sources", [])
            rays = self.fullGeometryData.get("fov_rays", [])
            trajectory = self.fullGeometryData.get("full_trajectory", [])
            bezier_curves = self.fullGeometryData.get("bezier_curves", [])

            total_frames = len(sources)
            if index < 0 or index >= total_frames:
                slicer.util.errorDisplay(f"Index {index} is out of bounds (max: {total_frames - 1}).")
                return

            self.clearModels()

            # Source point
            source = [sources[index]]
            self.createPointsModel(source, "Source", (0.2, 0.6, 1.0), 4)

            # Bézier surface
            if index < len(bezier_curves):
                bezier = bezier_curves[index]
                self.createBezierSurfaceModel(bezier, "DetectorBezier", (1.0, 0.5, 0.0), 0.7)
            else:
                print(f"[WARNING] Missing Bézier surface at index {index}")

            # FOV rays
            raw_rays = rays[index] if index < len(rays) else []
            ray_set = []
            for item in raw_rays:
                if isinstance(item, list) and len(item) == 2:
                    p1, p2 = item
                    ray_set.append((p1, p2))
                else:
                    print(f"[WARNING] Skipping malformed ray at index {index}: {item}")
            self.createLinesModel(ray_set, "FOVRays", (0.0, 1.0, 0.0), 2)

            # Trajectory line (only once)
            if index == 0 and trajectory:
                self.createTrajectoryLine(trajectory, "SourceTrajectory", (0.0, 0.4, 1.0), 2)

            # Update label
            self.currentIndex = index
            self.ui.sliderIndexLabel.setText(f"Index: {index}")
            print(f"[DEBUG] Rendering index {index} with {len(ray_set)} rays.")

        except Exception as e:
            slicer.util.errorDisplay(f"Failed to update frame {index}:\n{e}")

    # -----------------------------
    # Slider / sinogram fetch
    # -----------------------------
    def onIncreaseWindow(self):
        self.currentIndex += 20
        if hasattr(self, "fullGeometryData"):
            self.updateViewFromFullDataset(self.currentIndex)

    def onDecreaseWindow(self):
        self.currentIndex = max(0, self.currentIndex - 20)
        if hasattr(self, "fullGeometryData"):
            self.updateViewFromFullDataset(self.currentIndex)

    def onIndexChanged(self, value):
        self.currentIndex = value
        if hasattr(self.ui, "windowSizeLabel"):
            self.ui.windowSizeLabel.setText(f"Index: {value}")
        self.sliderDebounceTimer.start()

    def onFetchSinogramSliceClicked(self):
        try:
            index = self.currentIndex
            base = self._currentBaseUrl()
            url = f"{base}/get_sinogram_slice/{index}"
            filename = f"sinogram_{index}.nrrd"
            temp_path = os.path.join(slicer.app.temporaryPath, filename)

            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Use 'singleFile' option to avoid archetype warning
            volume_node = slicer.util.loadVolume(temp_path, {'singleFile': True})

            if volume_node:
                for view in ["Red", "Green", "Yellow"]:
                    slicer.app.layoutManager().sliceWidget(view).sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(volume_node.GetID())
                slicer.app.layoutManager().resetSliceViews()
            else:
                slicer.util.errorDisplay("Failed to load sinogram slice.")
        except Exception as e:
            slicer.util.errorDisplay(f"Error fetching sinogram slice: {e}")


# -----------------------------
# HTTP helpers (use saved base URL)
# -----------------------------
def stream_nrrd_from_url(filename, base_url: str = None):
    try:
        base = normalize_base_url(base_url or get_saved_base_url())
        url = f"{base}/images/{filename}"
        temp_file_path = os.path.join(slicer.app.temporaryPath, filename)

        response = requests.head(url)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        max_progress = min(total_size, 2**31 - 1)
        scale_factor = total_size / max_progress if total_size > max_progress else 1

        progress_dialog = QProgressDialog("Loading file...", "Cancel", 0, max_progress)
        progress_dialog.setWindowTitle("Load Progress")
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setValue(0)

        canceled = [False]
        progress_dialog.canceled.connect(lambda: canceled.__setitem__(0, True))

        with requests.get(url, stream=True) as r, open(temp_file_path, "wb") as temp_file:
            r.raise_for_status()
            downloaded_size = 0
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
                    downloaded_size += len(chunk)
                    progress_dialog.setValue(int(downloaded_size / scale_factor))
                    slicer.app.processEvents()
                    if canceled[0]:
                        slicer.util.errorDisplay("Download was canceled by the user.", windowTitle="Download Canceled")
                        return

        volume_node = slicer.util.loadVolume(temp_file_path)
        if volume_node:
            for view in ["Red", "Green", "Yellow"]:
                slicer.app.layoutManager().sliceWidget(view).sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(volume_node.GetID())
            slicer.app.layoutManager().resetSliceViews()
        else:
            print("Failed to load the NRRD volume.")

    except requests.exceptions.RequestException as e:
        slicer.util.errorDisplay(f"Failed to download the NRRD file from:\n{url}\n\nError: {e}", windowTitle="Download Failed")
    except Exception as e:
        slicer.util.errorDisplay(f"Failed to process or load the NRRD file.\n\nError: {e}", windowTitle="Processing Failed")


def stream_vtp_from_url(filenames, base_url: str = None):
    """
    Download and load one or more VTP files into 3D Slicer with appropriate coloring and settings.
    """
    base = normalize_base_url(base_url or get_saved_base_url())

    color_map = {
        "fov_rays.vtp": (0.0, 1.0, 0.0),     # Green
        "source_path.vtp": (0.0, 0.0, 1.0),  # Blue
        "detector_path.vtp": (1.0, 0.0, 0.0) # Red
    }

    for filename in filenames:
        try:
            url = f"{base}/{filename}"  # e.g. {base}/files/xyz.vtp if you place under /files
            temp_file_path = os.path.join(slicer.app.temporaryPath, filename)

            response = requests.get(url, timeout=10)
            response.raise_for_status()

            with open(temp_file_path, 'wb') as f:
                f.write(response.content)

            model_node = slicer.util.loadModel(temp_file_path)

            if model_node:
                display_node = model_node.GetDisplayNode()
                if display_node:
                    display_node.SetVisibility(1)
                    display_node.SetLineWidth(4)
                    display_node.SetScalarVisibility(False)

                    color = color_map.get(filename, (0.5, 0.5, 0.5))  # Default gray if unknown
                    display_node.SetColor(*color)

                    print(f"[INFO] Loaded {filename} with color {color}.")

                bounds = [0]*6
                model_node.GetBounds(bounds)
                print(f"[DEBUG] {filename} bounds: {bounds}")
            else:
                slicer.util.errorDisplay(f"Failed to load model: {filename}")

        except Exception as e:
            slicer.util.errorDisplay(f"Failed to load {filename}: {e}")

    layoutManager = slicer.app.layoutManager()
    threeDWidget = layoutManager.threeDWidget(0)
    threeDView = threeDWidget.threeDView()
    threeDView.resetFocalPoint()


class SinoReconsVisual2Logic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)


class SinoReconsVisual2Test(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.delayDisplay("Test not implemented.")