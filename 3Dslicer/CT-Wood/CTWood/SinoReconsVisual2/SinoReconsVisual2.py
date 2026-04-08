import os
import json
import logging
import requests
import qt
import slicer
import vtk
import time
import io
# This is needed for avif support in pillow. 
# 3D slicer needs to be restarted for the upgrade to properly load.
#slicer.util.pip_install("--upgrade pillow")
import PIL
import PIL.Image
import numpy as np
import typing
from typing import Optional
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy, numpy_to_vtkIdTypeArray
from vtkmodules.util.vtkImageImportFromArray import vtkImageImportFromArray
import gzip

try:
  import nrrd
except ModuleNotFoundError:
  if slicer.util.confirmOkCancelDisplay("This module requires 'pynrrd' Python package. Click OK to install it now."):
    slicer.util.pip_install("pynrrd")
    import nrrd

from urllib.parse import urlparse
from qt import QProgressDialog
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from qt import QTimer

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

def get_icons_folder() -> str:
    return os.path.join(os.path.dirname(__file__), "Resources", "Icons")

class SinoReconsVisual2(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("SinoReconsVisual2")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Utilities")]
        self.parent.dependencies = []
        self.parent.contributors = ["Oualid Burström (LTU)", "Julius Häger (KTH)"]
        self.parent.helpText = _("The SinoReconsVisual2 extension enables users ...")
        self.parent.acknowledgementText = _("")
        iconPath = os.path.join(get_icons_folder(), "SinoReconsVisual2.png")
        if os.path.exists(iconPath):
            self.parent.icon = qt.QIcon(iconPath)

class Vtk3DSceneObjects:
    sourceModel: slicer.vtkMRMLModelNode
    sourceModelDisplay: slicer.vtkMRMLModelDisplayNode

    fovRaysModel: slicer.vtkMRMLModelNode
    fovRaysModelDisplay: slicer.vtkMRMLModelDisplayNode

    sensorModel: slicer.vtkMRMLModelNode
    sensorModelDisplay: slicer.vtkMRMLModelDisplayNode
    sensorModelImage: vtk.vtkImageData
    sensorModelImageProducer: vtk.vtkTrivialProducer

    trajectoryModel: slicer.vtkMRMLModelNode
    trajectoryModelDisplay: slicer.vtkMRMLModelDisplayNode

    sinogramOutline: vtk.vtkOutlineSource
    sinogramOutlineNode: slicer.vtkMRMLModelNode
    sinogramOutlineDisplay: slicer.vtkMRMLModelDisplayNode

    def __init__(self):
        pass

class SampleData:
    totalSamples: int

    sinogram_min: np.float32
    sinogram_max: np.float32

    bounds_min: np.typing.NDArray[np.float32]
    bounds_max: np.typing.NDArray[np.float32]

    geometry: dict[str, np.typing.NDArray]

    def __init__(self):
        pass

class ROIData:
    roi_node: slicer.vtkMRMLMarkupsROINode
    sinogram_start_index: int
    sinogram_end_index: int

class SinoReconsVisual2Widget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    sceneObjects: Vtk3DSceneObjects
    sampleData: SampleData

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.sampleData = SampleData()

    # -----------------------------
    # UI & wiring
    # -----------------------------
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        self.session = requests.Session()

        # Load and attach UI
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SinoReconsVisual2.ui"))
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        self.layout.addWidget(uiWidget)

        self.registerSinogramLayout()

        # Initialize saved backend URL into the line edit (new field in UI)
        if hasattr(self.ui, "serverUrlLineEdit") and self.ui.serverUrlLineEdit is not None:
            self.ui.serverUrlLineEdit.setText(get_saved_base_url())

        # Initialize window size
        self.windowSize = 1
        self.currentIndex = 0

        # Connect signals to slots
        self.ui.indexSlider.valueChanged.connect(self.onIndexChanged)

        self.ui.connectToServerButton.clicked.connect(self.onConnectToServerClicked)
        self.ui.loadSampleButton.clicked.connect(self.loadSelectedSample)

        self.ui.metadataGroupBox.visible = False
        self.ui.metadataGroupBox.collapsed = True
        self.ui.metadataTableWidget.setEditTriggers(0) # No editing

        self.ui.runReconstructionButton.clicked.connect(self.onRunReconstructionClicked)
        self.ui.loadReconstructionDataButton.clicked.connect(self.onLoadReconstructionDataClicked)

        # FIXME: Change to currentIndexChanged...
        self.ui.reconstructionMethodComboBox.currentTextChanged.connect(self.onReconstructionMethodChanged)
        self.ui.showSinogramOnSensorCheckbox.stateChanged.connect(self.setImageOnSensor)

        self.ui.playButton.toggled.connect(self.playButtonToggled)
        self.ui.playSpeedSlider.valueChanged.connect(lambda x : self.ui.playSpeedLabel.setText(f"Speed: x{x}"))
        self.playButtonTimer = QTimer()
        self.playButtonTimer.setInterval(10)
        self.playButtonTimer.timeout.connect(self.advanceSinogramSlice)

        # Slider debounce timer
        self.sliderDebounceTimer = QTimer()
        self.sliderDebounceTimer.setSingleShot(True)
        self.sliderDebounceTimer.setInterval(1)  # milliseconds
        self.sliderDebounceTimer.timeout.connect(self.loadPreviewSlice)

        # Timer to debounce loading full detail (float32) slice
        self.fullDetailDebounceTimer = QTimer()
        self.fullDetailDebounceTimer.setSingleShot(True)
        self.fullDetailDebounceTimer.setInterval(500)
        self.fullDetailDebounceTimer.timeout.connect(self.loadFullDetailSlice)

        # FIXME: For some reason it seems like we can't do this here
        # because slicer.mrmlScene isn't set here or something similar.
        # - Julius Häger 2026-03-05
        #self.sceneObjects = self.createSceneObjects()

        self.ui.roiCenterCoordinateWidget.coordinatesChanged.connect(self.roiCenterChanged)
        self.ui.roiSizeCoordinateWidget.coordinatesChanged.connect(self.roiSizeChanged)
        self.ui.sinogramRangeWidget.valuesChanged.connect(self.roiSinogramValuesChanged)

        self.ui.addROIButton.clicked.connect(self.addROI)
        self.ui.removeROIButton.clicked.connect(self.removeROI)
        self.ui.roiListWidget.currentItemChanged.connect(self.selectROI)
        self.ui.roiListWidget.itemChanged.connect(self.roiItemChanged)
        self.selectROI(None, None)
        self.ui.roiListWidgetEventFilter = self.ROIListEventFilter(self.ui.roiListWidget)

    class ROIListEventFilter(qt.QObject):
        roi_list_widget: qt.QListWidget
        roi_list_widget_viewport: typing.Any

        def __init__(self, roi_list_widget: qt.QListWidget):
            qt.QObject.__init__(self)
            self.roi_list_widget = roi_list_widget
            self.roi_list_widget.installEventFilter(self)

            self.roi_list_widget_viewport = self.roi_list_widget.viewport()
            self.roi_list_widget_viewport.installEventFilter(self)
        
        def eventFilter(self, source, event) -> bool:
            # FIXME: InputMethodQueryEvents cause all kinds of havock crashing 3D Slicer all together.
            # The crashing only happens sometimes and it seems to be related to recursive calls to eventFilter
            # though I have no idea how this function is getting called recursively...
            # - Julius Häger 2026-04-08
            if event.type() == qt.QEvent.InputMethodQuery:
                return False
            
            if source is self.roi_list_widget:
                if event.type() == qt.QEvent.KeyPress:
                    if event.key() == qt.Qt.Key_Escape:
                        self.roi_list_widget.selectionModel().clear()
                        self.roi_list_widget.setCurrentRow(-1)
            elif source is self.roi_list_widget_viewport:
                if event.type() == qt.QEvent.MouseButtonPress:
                    if self.roi_list_widget.indexAt(event.pos()).isValid() == False:
                        self.roi_list_widget.selectionModel().clear()
                        self.roi_list_widget.setCurrentRow(-1)

            return False

    def registerSinogramLayout(self):
        layoutDesc = """
        <layout type="vertical">
            <item><view class="vtkMRMLViewNode" singletontag="1">
                <property name="viewlabel" action="default">1</property>
                <property name="viewcolor" action="default">#FFFF00</property>
            </view></item>
            <item><view class="vtkMRMLSliceNode" singletontag="Yellow">
                <property name="orientation" action="default">Axial</property>
                <property name="viewlabel" action="default">Y</property>
                <property name="viewcolor" action="default">#0000FF</property>
            </view></item>
        </layout>
        """
        customLayoutId = 1234

        # FIXME: Unload layout desc when we reload the plugin!
        layoutManager = slicer.app.layoutManager()
        if not layoutManager.layoutLogic().GetLayoutNode().SetLayoutDescription(customLayoutId, layoutDesc):
            layoutManager.layoutLogic().GetLayoutNode().AddLayoutDescription(customLayoutId, layoutDesc)
            layoutManager.setLayout(customLayoutId)

            slicer.app.layoutManager().threeDWidget(0).mrmlAbstractViewNode().SetFPSVisible(True)

            # Add button to layout selector toolbar for this custom layout
            viewToolBar = slicer.util.mainWindow().findChild("QToolBar", "ViewToolBar")
            layoutMenu = viewToolBar.widgetForAction(viewToolBar.actions()[0]).menu()
            layoutSwitchActionParent = layoutMenu  # use `layoutMenu` to add inside layout list, use `viewToolBar` to add next the standard layout list
            layoutSwitchAction = layoutSwitchActionParent.addAction("Sinogram layout") # add inside layout list
            layoutSwitchAction.setData(customLayoutId)
            layoutSwitchAction.setIcon(qt.QIcon(os.path.join(get_icons_folder(), "SinoReconsVisual2.png")))
            layoutSwitchAction.setToolTip("3D and sinogram view")

    def cleanup(self):
        self.destroySceneObjects()
        if hasattr(self, 'volume_node') and self.volume_node:
            slicer.mrmlScene.RemoveNode(self.volume_node)
        if hasattr(self, 'full_detail_volume_node') and self.full_detail_volume_node:
            slicer.mrmlScene.RemoveNode(self.full_detail_volume_node)
        
        for row in range(self.ui.roiListWidget.count):
            item = self.ui.roiListWidget.item(row)
            itemData: ROIData = item.data(0x0100)
            slicer.mrmlScene.RemoveNode(itemData.roi_node)

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

    def onLoadReconstructionDataClicked(self):
        try:
            tree_ID, disk_ID = self._parse_selected_sample()
            method = (self.ui.reconstructionSelectorComboBox.currentText or "").lower()
            filename = f"tree{tree_ID}_disk{disk_ID}_{method}.nrrd"
            base = self._currentBaseUrl()
            stream_nrrd_from_url(filename, base)
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to parse sample.\nError: {e}", windowTitle="Parsing Error")

    def onRunReconstructionClicked(self):
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
        method = (self.ui.reconstructionMethodComboBox.currentText or "").lower()
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
            import re, json as _json
            slicer.util.infoDisplay("Starting remote reconstruction…", "Reconstruction")

            r = self.session.post(run_url, json=payload, stream=True, timeout=None)
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
            response = self.session.get(config_url, timeout=5)
            response.raise_for_status()
            config = response.json()

            # Populate container/volume for completeness (from config)
            #container_name = config.get("container_name", "")
            #self.ui.containerSelectorComboBox.clear()
            #if container_name:
            #    self.ui.containerSelectorComboBox.addItem(container_name)

            #volume_name = config.get("volume_name", "")
            #self.ui.volumeSelectorComboBox.clear()
            #if volume_name:
            #    self.ui.volumeSelectorComboBox.addItem(volume_name)

            # Populate sample combo box
            samples = config.get("samples", [])
            self.ui.sampleSelectorComboBox.clear()
            for sample in samples:
                # expect keys: specie, tree_ID, disk_ID
                sample_text = f"Tree {sample['tree_ID']} - Disk {sample['disk_ID']}"
                self.ui.sampleSelectorComboBox.addItem(sample_text, sample)
            self.ui.selectSampleLabel.setEnabled(True)
            self.ui.sampleSelectorComboBox.setEnabled(True)
            self.ui.loadSampleButton.setEnabled(True)

            # Populate reconstruction methods
            recons = config.get("reconstruction_methods", [])
            self.ui.reconstructionMethodComboBox.clear()
            for r in recons:
                self.ui.reconstructionMethodComboBox.addItem(r)

            # Initial visibility for parameter groups
            selected_method = self.ui.reconstructionMethodComboBox.currentText
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

            self.ui.statusLabel.setText('Status: <font color="green">Connected</font>')

        except requests.exceptions.RequestException as e:
            self.ui.statusLabel.setText('Status: <font color="red">Failed to connect</font>')
            slicer.util.errorDisplay(f"Could not fetch config from:\n{config_url}\n\n{e}", windowTitle="Connection Error")

    def loadSelectedSample(self):
        try:
            self.destroySceneObjects()
            base = self._currentBaseUrl()

            sample = self.ui.sampleSelectorComboBox.currentData
            print(f"[INFO] Selecting sample: {sample["specie"]} {sample["tree_ID"]} {sample["disk_ID"]}")

            url = f"{base}/select_sample"
            start = time.time()
            payload = { "specie": sample["specie"], "tree_ID": int(sample["tree_ID"]), "disk_ID": int(sample["disk_ID"]) }
            response = self.session.post(url, json=payload, timeout=300)  # Up to 5 minutes
            if response.status_code != 200:
                self.ui.sinogramWidget.setEnabled(False)
                self.ui.reconstructionWidget.setEnabled(False)
                self.ui.roiWidget.setEnabled(False)
                self.ui.statusLabel.setText('Status: <font color="red">Failed to load sample</font>')
                slicer.util.errorDisplay(f"Failed to load sample {response.status_code}\n{response.content}")
            response_json = response.json()
            self.sampleData.sinogram_min = response_json["sinogram_min"]
            self.sampleData.sinogram_max = response_json["sinogram_max"]
            end = time.time()
            print(f"[INFO] Selecting sample took: {end-start} seconds")

            start = time.time()
            url = f"{base}/full_geometry_npz"
            response = self.session.get(url, timeout=300)  # Up to 5 minutes
            response.raise_for_status()
            self.sampleData.geometry = np.load(io.BytesIO(response.content), allow_pickle=False)
            self.sampleData.totalSamples = len(self.sampleData.geometry.get("sources", []))
            self.sampleData.bounds_max = np.max(self.sampleData.geometry.get("sources", np.empty(0)), axis = 0)
            self.sampleData.bounds_min = np.min(self.sampleData.geometry.get("sources", np.empty(0)), axis = 0)
            end = time.time()
            print(f"[INFO] Loaded full geometry: "
                  f"{len(self.sampleData.geometry.get("sources", []))} sources, "
                  f"{len(self.sampleData.geometry.get('detector_panels', []))} panels "
                  f"in {end - start} seconds {len(response.content)} bytes")

            # Configure slider
            self.ui.indexSlider.setMinimum(0)
            self.ui.indexSlider.setMaximum(max(0, self.sampleData.totalSamples - 1))
            self.ui.indexSlider.setValue(0)
            self.ui.sliderIndexLabel.setText("Index: 0")

            # Load first frame
            self.sceneObjects = self.createSceneObjects()
            self.onIndexChanged(0)
            self.setImageOnSensor(self.ui.showSinogramOnSensorCheckbox.checkState())

            self.ui.sinogramWidget.setEnabled(True)
            self.ui.reconstructionWidget.setEnabled(True)
            self.ui.roiWidget.setEnabled(True)
            self.ui.sinogramRangeWidget.setRange(0, self.sampleData.totalSamples)

            sample_metadata = response_json["metadata"]

            self.ui.metadataGroupBox.visible = True
            self.ui.metadataTableWidget.clear()
            self.ui.metadataTableWidget.setRowCount(0)
            self.ui.metadataTableWidget.setHorizontalHeaderLabels(["Name", "Value"])
            self.ui.metadataTableWidget.sortingEnabled = False
            for key, value in sample_metadata.items():
                self.ui.metadataTableWidget.insertRow(0)
                nameItem = qt.QTableWidgetItem()
                nameItem.setText(key)
                self.ui.metadataTableWidget.setItem(0, 0, nameItem)
                valueItem = qt.QTableWidgetItem()
                valueItem.setText(value)
                self.ui.metadataTableWidget.setItem(0, 1, valueItem)
            self.ui.metadataTableWidget.sortingEnabled = True

            # So that the units get set correctly
            self.ui.roiCenterCoordinateWidget.setMRMLScene(slicer.mrmlScene)
            self.ui.roiSizeCoordinateWidget.setMRMLScene(slicer.mrmlScene)

        except Exception as e:
            slicer.util.errorDisplay(f"Failed to load full dataset:\n{e}")

    def addROI(self):
        name = "roi 1"

        roi_data = ROIData()
        roi_data.roi_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        roi_data.roi_node.GetMarkupsDisplayNode().Visibility2DOff()
        roi_data.roi_node.SetName(name)

        # Update UI with the new roi
        roi_data.roi_node.AddObserver(vtk.vtkCommand.ModifiedEvent, self.roiNodeChanged)

        roi_data.sinogram_start_index = 0
        roi_data.sinogram_end_index = self.sampleData.totalSamples

        item = qt.QListWidgetItem()
        # FIXME: Find an free name
        item.setText(name)
        item.setFlags(2 + 32) # Editable + Enabled
        # FIXME: Change dict to object
        item.setData(0x0100, roi_data) # UserRole

        self.setInteractive(item, False)

        self.ui.roiListWidget.addItem(item)
        self.ui.roiListWidget.setCurrentItem(item)

    def roiItemChanged(self, item: qt.QListWidgetItem):
        #itemData: ROIData = item.data(0x0100)
        #itemData.roi_node.SetName(item.text())
        pass

    def removeROI(self):
        if self.ui.roiListWidget.currentRow != -1:
            item = self.ui.roiListWidget.takeItem(self.ui.roiListWidget.currentRow)
            itemData: ROIData = item.data(0x0100)
            slicer.mrmlScene.RemoveNode(itemData.roi_node)
        
    def selectROI(self, current: qt.QListWidgetItem, previous: qt.QListWidgetItem):
        if current == None:
            self.ui.roiEditWidget.setEnabled(False)
            self.ui.removeROIButton.setEnabled(False)
        else:
            self.ui.roiEditWidget.setEnabled(True)
            self.ui.removeROIButton.setEnabled(True)

            itemData: ROIData = current.data(0x0100)
            roiNode = itemData.roi_node

            center = roiNode.GetCenter()
            size = roiNode.GetSize()

            # FIXME: Very inefficient to set the coordinates through strings
            self.ui.roiCenterCoordinateWidget.coordinates = f"{center.GetX()},{center.GetY()},{center.GetZ()}"
            self.ui.roiSizeCoordinateWidget.coordinates = f"{size[0]},{size[1]},{size[2]}"

            self.ui.sinogramRangeWidget.setValues(itemData.sinogram_start_index, itemData.sinogram_end_index)

        if current != None:
            self.setInteractive(current, True)

        if previous != None:
            self.setInteractive(previous, False)

    def setInteractive(self, listWidgetItem: qt.QListWidgetItem, enable: bool):
        itemData: ROIData = listWidgetItem.data(0x0100)
        roiNode = itemData.roi_node
        roiDisplayNode = roiNode.GetMarkupsDisplayNode()
        if enable:
            roiDisplayNode.SetSelectedColor(1.0, 0.0, 0.0)
            roiDisplayNode.SetHandlesInteractive(True)
            roiDisplayNode.SetFillOpacity(0.7)
            roiDisplayNode.SetOutlineOpacity(1.0)
            roiDisplayNode.SetTranslationHandleVisibility(True)
            roiDisplayNode.SetRotationHandleVisibility(True)
            roiDisplayNode.SetScaleHandleVisibility(True)
        else:
            roiDisplayNode.SetSelectedColor(0.1, 0.1, 0.1)
            roiDisplayNode.SetHandlesInteractive(False)
            roiDisplayNode.SetFillOpacity(0.2)
            roiDisplayNode.SetOutlineOpacity(0.4)

    # PythonQt seems to bind the parameter as a 'double'
    # instead of a 'double*' meaning the parameter is
    # completely useless in python.
    # - Julius Häger 2026-04-07
    def roiCenterChanged(self, _broken):
        if self.ui.roiListWidget.currentRow == -1:
            return
        
        newCoords = [float(x) for x in self.ui.roiCenterCoordinateWidget.coordinates.split(',')]

        item = self.ui.roiListWidget.item(self.ui.roiListWidget.currentRow)
        itemData: ROIData = item.data(0x0100)
        roiNode = itemData.roi_node

        roiNode.SetCenter(newCoords)

    def roiSizeChanged(self, _broken):
        if self.ui.roiListWidget.currentRow == -1:
            return
        
        newSize = [float(x) for x in self.ui.roiSizeCoordinateWidget.coordinates.split(',')]

        item = self.ui.roiListWidget.item(self.ui.roiListWidget.currentRow)
        itemData: ROIData = item.data(0x0100)
        roiNode = itemData.roi_node

        roiNode.SetSize(newSize)

    # Called when the roi is changed using the 3D/2D widgets.
    def roiNodeChanged(self, roiNode, event):
        # FIXME: Only change the UI if this is the active roiNode?
        center = roiNode.GetCenter()
        size = roiNode.GetSize()

        # FIXME: Very inefficient to set the coordinates through strings
        self.ui.roiCenterCoordinateWidget.blockSignals(True)
        self.ui.roiCenterCoordinateWidget.coordinates = f"{center.GetX()},{center.GetY()},{center.GetZ()}"
        self.ui.roiCenterCoordinateWidget.blockSignals(False)

        self.ui.roiSizeCoordinateWidget.blockSignals(True)
        self.ui.roiSizeCoordinateWidget.coordinates = f"{size[0]},{size[1]},{size[2]}"
        self.ui.roiSizeCoordinateWidget.blockSignals(False)

    def roiSinogramValuesChanged(self, minVal: float, maxVal: float):
        if self.ui.roiListWidget.currentRow == -1:
            return

        minVal = int(minVal)
        maxVal = int(maxVal)

        item = self.ui.roiListWidget.item(self.ui.roiListWidget.currentRow)
        itemData: ROIData = item.data(0x0100)
        itemData.sinogram_start_index = minVal
        itemData.sinogram_end_index = maxVal

    # -----------------------------
    # Rendering helpers
    # -----------------------------
    def createSceneObjects(self):
        sceneObjects = Vtk3DSceneObjects()

        sceneObjects.sourceModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "Source")
        polys = vtk.vtkPolyData()
        polys.SetPoints(vtk.vtkPoints())
        polys.SetVerts(vtk.vtkCellArray())
        sceneObjects.sourceModel.SetAndObservePolyData(polys)
        sceneObjects.sourceModelDisplay = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        sceneObjects.sourceModel.SetAndObserveDisplayNodeID(sceneObjects.sourceModelDisplay.GetID())
        sceneObjects.sourceModelDisplay.SetColor((1.0, 1.0, 0.0))
        sceneObjects.sourceModelDisplay.SetPointSize(8)
        sceneObjects.sourceModelDisplay.SetVisibility(1)

        sceneObjects.fovRaysModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "FOVRays")
        polys = vtk.vtkPolyData()
        polys.SetPoints(vtk.vtkPoints())
        polys.SetLines(vtk.vtkCellArray())
        sceneObjects.fovRaysModel.SetAndObservePolyData(polys)
        sceneObjects.fovRaysModelDisplay = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        sceneObjects.fovRaysModel.SetAndObserveDisplayNodeID(sceneObjects.fovRaysModelDisplay.GetID())
        sceneObjects.fovRaysModelDisplay.SetColor((0.0, 1.0, 0.0))
        sceneObjects.fovRaysModelDisplay.SetLineWidth(2)
        sceneObjects.fovRaysModelDisplay.SetVisibility(1)

        sceneObjects.sensorModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "Detector")
        polys = vtk.vtkPolyData()
        polys.SetPoints(vtk.vtkPoints())
        polys.SetPolys(vtk.vtkCellArray())
        sceneObjects.sensorModel.SetAndObservePolyData(polys)
        sceneObjects.sensorModelDisplay = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        sceneObjects.sensorModel.SetAndObserveDisplayNodeID(sceneObjects.sensorModelDisplay.GetID())
        sceneObjects.sensorModelDisplay.SetColor((1.0, 0.5, 0.0))
        sceneObjects.sensorModelDisplay.SetOpacity(0.7)
        sceneObjects.sensorModelDisplay.SetVisibility(1)
        sceneObjects.sensorModelDisplay.SetRepresentation(2)          # Surface
        sceneObjects.sensorModelDisplay.SetEdgeVisibility(False)      # Hide edges
        sceneObjects.sensorModelDisplay.SetLighting(0)                # Disable lighting
        sceneObjects.sensorModelDisplay.SetBackfaceCulling(False)     # Render both sides
        sceneObjects.sensorModelImage = vtk.vtkImageData()
        sceneObjects.sensorModelImageProducer = vtk.vtkTrivialProducer()
        sceneObjects.sensorModelImageProducer.SetOutput(sceneObjects.sensorModelImage)

        sceneObjects.trajectoryModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "Trajectory")
        polys = vtk.vtkPolyData()
        polys.SetPoints(vtk.vtkPoints())
        polys.SetLines(vtk.vtkCellArray())
        sceneObjects.trajectoryModel.SetAndObservePolyData(polys)
        sceneObjects.trajectoryModelDisplay = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        sceneObjects.trajectoryModel.SetAndObserveDisplayNodeID(sceneObjects.trajectoryModelDisplay.GetID())
        sceneObjects.trajectoryModelDisplay.SetColor((0.0, 0.4, 1.0))
        sceneObjects.trajectoryModelDisplay.SetLineWidth(2)
        sceneObjects.trajectoryModelDisplay.SetVisibility(1)

        yellowViewNodeID = slicer.app.layoutManager().sliceWidget("Yellow").mrmlSliceNode().GetID()

        # For the sinogram outline we project a cube outline to the slice only in the Yellow view.
        # - Julius Häger 2026-03-27
        sceneObjects.sinogramOutline = vtk.vtkOutlineSource()
        sceneObjects.sinogramOutline.SetBounds(-3, 3, 0, 50, 0, 50)
        sceneObjects.sinogramOutlineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "Sinogram Outline")
        sceneObjects.sinogramOutlineNode.SetPolyDataConnection(sceneObjects.sinogramOutline.GetOutputPort())
        sceneObjects.sinogramOutlineDisplay = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        sceneObjects.sinogramOutlineNode.SetAndObserveDisplayNodeID(sceneObjects.sinogramOutlineDisplay.GetID())
        sceneObjects.sinogramOutlineDisplay.SetViewNodeIDs([yellowViewNodeID])
        sceneObjects.sinogramOutlineDisplay.SetSliceDisplayModeToProjection()
        sceneObjects.sinogramOutlineDisplay.Visibility3DOff()
        sceneObjects.sinogramOutlineDisplay.Visibility2DOn()

        return sceneObjects

    def destroySceneObjects(self):
        if (hasattr(self, 'sceneObjects')) and self.sceneObjects:
            slicer.mrmlScene.RemoveNode(self.sceneObjects.sourceModelDisplay)
            slicer.mrmlScene.RemoveNode(self.sceneObjects.sourceModel)
            slicer.mrmlScene.RemoveNode(self.sceneObjects.fovRaysModelDisplay)
            slicer.mrmlScene.RemoveNode(self.sceneObjects.fovRaysModel)
            slicer.mrmlScene.RemoveNode(self.sceneObjects.sensorModelDisplay)
            slicer.mrmlScene.RemoveNode(self.sceneObjects.sensorModel)
            slicer.mrmlScene.RemoveNode(self.sceneObjects.trajectoryModelDisplay)
            slicer.mrmlScene.RemoveNode(self.sceneObjects.trajectoryModel)
            slicer.mrmlScene.RemoveNode(self.sceneObjects.sinogramOutlineDisplay)
            slicer.mrmlScene.RemoveNode(self.sceneObjects.sinogramOutlineNode)
            self.sceneObjects.sourceModelDisplay = None
            self.sceneObjects.sourceModel = None
            self.sceneObjects.fovRaysModelDisplay = None
            self.sceneObjects.fovRaysModel = None
            self.sceneObjects.sensorModelDisplay = None
            self.sceneObjects.sensorModel = None
            self.sceneObjects.trajectoryModelDisplay = None
            self.sceneObjects.trajectoryModel = None
            self.sceneObjects.sinogramOutlineNode = None
            self.sceneObjects.sinogramOutlineDisplay = None

    def updateSceneData(self, index: int):
        self.updateTrajectory()
        self.updateSource(index)
        self.updateFOVLines(index)
        self.updateSensorGeometry(index)

    def updateTrajectory(self):
        trajectory = self.sampleData.geometry.get("full_trajectory", np.empty(0))
        #print(f"trajectory {trajectory.shape} {trajectory.dtype} {len(trajectory)}")
        
        polyData : vtk.vtkPolyData = typing.cast(vtk.vtkPolyData, self.sceneObjects.trajectoryModel.GetPolyData())

        points = typing.cast(vtk.vtkPoints, polyData.GetPoints())
        points.SetData(numpy_to_vtk(trajectory, 1))

        cells = polyData.GetLines()
        idx = numpy_to_vtkIdTypeArray(np.arange(len(trajectory)), 1)
        cells.SetData(len(trajectory), idx)

        points.Modified()
        cells.Modified()
        polyData.Modified()

    def updateSource(self, index: int):
        sources = self.sampleData.geometry.get("sources", np.empty(0))
        source = sources[index].reshape(1, 3)
        #print(f"sources {source.shape} {source.dtype} {len(source)}")

        polyData : vtk.vtkPolyData = typing.cast(vtk.vtkPolyData, self.sceneObjects.sourceModel.GetPolyData())
        
        points = typing.cast(vtk.vtkPoints, polyData.GetPoints())
        points.SetData(numpy_to_vtk(source, 1))
        #print(points)

        cells = polyData.GetVerts()
        idx = numpy_to_vtkIdTypeArray(np.arange(len(source)), 1)
        cells.SetData(len(source), idx)
        #print(cells)

        points.Modified()
        cells.Modified()
        polyData.Modified()
        #print(polyData)

    def updateFOVLines(self, index: int):
        rays = self.sampleData.geometry.get("fov_rays", np.empty(0))
        rays = rays[index]
        if rays.shape[-1] != 3 or rays.shape[-2] != 2:
            print(f"[ERROR] fov_rays had the wrong shape {rays.shape}, expected (N, 2, 3).")
            return
        
        polyData : vtk.vtkPolyData = typing.cast(vtk.vtkPolyData, self.sceneObjects.fovRaysModel.GetPolyData())
        
        points = typing.cast(vtk.vtkPoints, polyData.GetPoints())
        points.SetData(numpy_to_vtk(np.array(rays).reshape(len(rays)*2, 3), 1))
        #print(points)

        cells = polyData.GetLines()
        cells.SetData(2, numpy_to_vtkIdTypeArray(np.arange(len(rays)*2), 1))
        #print(cells)

        points.Modified()
        cells.Modified()
        polyData.Modified()

    def updateSensorGeometry(self, index: int):
        bezier_curves = self.sampleData.geometry.get("bezier_curves", np.empty(0))
        bezier_curves_uvs = self.sampleData.geometry.get("bezier_curves_uvs", np.empty(0))
        curves = np.array(bezier_curves[index])
        curve_uvs = np.array(bezier_curves_uvs)

        num_rows = len(curves)
        num_cols = len(curves[0])
        if any(len(row) != num_cols for row in curves):
            print("[ERROR] Inconsistent number of points per curve row.")
            return
        
        if len(curve_uvs) != num_rows or any(len(row) != num_cols for row in curve_uvs):
            print("[ERROR] Inconsistent number of points per curve uv row.")
            return
        #print(f"sensor {num_rows} {num_cols} {curves.shape}")
        
        def idx(i: int, j: int):
            return i * num_cols + j

        indices = np.zeros((num_rows - 1, num_cols - 1, 4), dtype=np.int64)
        for i in range(num_rows - 1):
            for j in range(num_cols - 1):
                indices[i, j, 0] = idx(i, j)
                indices[i, j, 1] = idx(i, j + 1)
                indices[i, j, 2] = idx(i + 1, j + 1)
                indices[i, j, 3] = idx(i + 1, j)

        polyData : vtk.vtkPolyData = typing.cast(vtk.vtkPolyData, self.sceneObjects.sensorModel.GetPolyData())
        
        points = typing.cast(vtk.vtkPoints, polyData.GetPoints())
        points.SetData(numpy_to_vtk(curves.reshape(num_rows * num_cols, 3), 1))
        #print(points)

        pointData = polyData.GetPointData()
        pointData.SetTCoords(numpy_to_vtk(curve_uvs.reshape(num_rows * num_cols, 2), 1))
        #print(pointData)

        cells = polyData.GetPolys()
        cells.SetData(4, numpy_to_vtkIdTypeArray(np.ravel(indices), 1))
        #print(cells)

        points.Modified()
        pointData.Modified()
        cells.Modified()
        polyData.Modified()

        #print(polyData)

    def setImageOnSensor(self, state : int):
        if hasattr(self, "sceneObjects") == False or self.sceneObjects == None:
            return
        
        print(f"show on sensor: {state}")
        if state == 2:
            self.sceneObjects.sensorModelDisplay.SetTextureImageDataConnection(self.sceneObjects.sensorModelImageProducer.GetOutputPort())
            self.sceneObjects.sensorModelDisplay.SetOpacity(1.0)
            self.sceneObjects.sensorModelDisplay.SetInterpolation(0) # Flat
            self.sceneObjects.sensorModelDisplay.SetScalarRangeFlag(1) # Data range = auto
        else:
            self.sceneObjects.sensorModelDisplay.SetTextureImageDataConnection(None)
            self.sceneObjects.sensorModelDisplay.SetOpacity(0.7)
            self.sceneObjects.sensorModelDisplay.SetInterpolation(1) # Phong

    # -----------------------------
    # Slider / sinogram fetch
    # -----------------------------
    def onIndexChanged(self, value):
        self.currentIndex = value
        self.ui.sliderIndexLabel.setText(f"Index: {value}")
        start = time.time()
        self.updateSceneData(value)
        end = time.time()
        print(f"[DEBUG] updateSceneData took {end - start} seconds")
        self.sliderDebounceTimer.start()
        self.fullDetailDebounceTimer.start()

    def loadPreviewSlice(self):
        try:
            index = self.currentIndex
            base = self._currentBaseUrl()

            start = time.time()
            response_fast = self.session.get(f"{base}/get_sinogram_slice_fast/{index}")
            end = time.time()
            print(f"[DEBUG] Download (fast) took {end-start} s {len(response_fast.content)/1000} kb")

            sliceMin = np.float32(response_fast.headers["slice_min"])
            sliceMax = np.float32(response_fast.headers["slice_max"])

            start = time.time()
            img = PIL.Image.open(io.BytesIO(response_fast.content))
            img_data = np.array(img)
            img_data_mapped = img_data * ((sliceMax - sliceMin) / np.float32(255.0)) + sliceMin
            end = time.time()
            print(f"[DEBUG] Decoding image took {end-start} s {img.size} {img.mode} {img_data_mapped.shape} {img_data_mapped.dtype}")

            start = time.time()
            if hasattr(self, 'volume_node') and self.volume_node:
                voxel_data = slicer.util.arrayFromVolume(self.volume_node)
                #print(f"voxel_data shape: {voxel_data.shape} {voxel_data.dtype}")
                voxel_data[:,:,0] = img_data_mapped
                slicer.util.arrayFromVolumeModified(self.volume_node)
            else:
                self.volume_node = slicer.util.addVolumeFromArray(img_data_mapped[:, :, np.newaxis])
                self.volume_node.name = f"Preview sinogram"

            self.volume_node.GetDisplayNode().SetAutoWindowLevel(False)
            self.volume_node.GetDisplayNode().SetWindowLevelMinMax(self.sampleData.sinogram_min, self.sampleData.sinogram_max)

            # Update outline bounds.
            bounds = np.empty(6)
            self.volume_node.GetBounds(bounds)
            self.sceneObjects.sinogramOutline.SetBounds(*bounds)

            end = time.time()
            print(f"[DEBUG] Adding/modifying volume took: {end - start} s")

            start = time.time()
            tex_data = (np.iinfo(np.uint8).max * (img_data_mapped - self.sampleData.sinogram_min) / (self.sampleData.sinogram_max - self.sampleData.sinogram_min)).astype(np.uint8)

            self.sceneObjects.sensorModelImage.SetDimensions(tex_data.shape[1], tex_data.shape[0], 1)
            # FIXME: Do not allocate a new VTK arrray, update the existing one if possible!
            vtk_array = numpy_to_vtk(tex_data.reshape(tex_data.shape[0] * tex_data.shape[1], 1), 1, vtk.VTK_UNSIGNED_CHAR)
            #print(vtk_array)
            self.sceneObjects.sensorModelImage.GetPointData().SetScalars(vtk_array)
            end = time.time()
            print(f"[DEBUG] Updating sensor texture took: {end - start} s")

            if self.volume_node:
                for view in ["Red", "Green", "Yellow"]:
                    slicer.app.layoutManager().sliceWidget(view).sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(self.volume_node.GetID())
                slicer.app.layoutManager().resetSliceViews()
            else:
                slicer.util.errorDisplay("Failed to load sinogram slice.")

        except Exception as e:
            self.sliderDebounceTimer.stop()
            import traceback
            #print(e)
            print(traceback.format_exc())
            #slicer.util.errorDisplay(f"Error fetching sinogram slice: {e} {traceback.format_exc()}")

    def loadFullDetailSlice(self):
        try:
            index = self.currentIndex
            base = self._currentBaseUrl()

            start = time.time()
            response = self.session.get(f"{base}/get_sinogram_slice/{index}")
            end = time.time()
            print(f"[DEBUG] Download (full detail) took {end-start} s {len(response.content)/1000} kb")

            start = time.time()
            bytes = io.BytesIO(response.content)
            header = nrrd.read_header(bytes)
            data = nrrd.read_data(header=header, fh=bytes)
            end = time.time()
            print(f"[DEBUG] Decoding image took {end-start} s {data.shape} {data.dtype}")

            #slice_2d = data.squeeze()
            #min = np.min(slice_2d)
            #max = np.max(slice_2d)
            #img_data = np.iinfo(np.uint8).max * (slice_2d - min) / (max - min)
            #im = PIL.Image.fromarray(img_data.astype(np.uint8))
            #im.save(f"C:/Users/juliu/Documents/SlicerCapture/{index}.jp2", format="", irreversible=True, quality_mode="dB", quality_layers=[44])
            #im.save(f"C:/Users/juliu/Documents/SlicerCapture/{index}_q57_s10.avif", format="", max_threads=1, quality=57, speed=10)
            #im.save(f"C:/Users/juliu/Documents/SlicerCapture/{index}_q57_s8.avif", format="", max_threads=1, quality=57, speed=8)
            #im.save(f"C:/Users/juliu/Documents/SlicerCapture/{index}_q57_s6.avif", format="", max_threads=1, quality=57, speed=6)
            #im.save(f"C:/Users/juliu/Documents/SlicerCapture/{index}_q57_s4.avif", format="", max_threads=1, quality=57, speed=4)
            #im.save(f"C:/Users/juliu/Documents/SlicerCapture/{index}_q57_s2.avif", format="", max_threads=1, quality=57, speed=2)
            #im.save(f"C:/Users/juliu/Documents/SlicerCapture/{index}_q57_s0.avif", format="", max_threads=1, quality=57, speed=0)

            start = time.time()
            if hasattr(self, 'full_detail_volume_node') and self.full_detail_volume_node:
                voxel_data = slicer.util.arrayFromVolume(self.full_detail_volume_node)
                #print(f"voxel_data shape: {voxel_data.shape} {voxel_data.dtype}")
                voxel_data[:,:,:] = data
                slicer.util.arrayFromVolumeModified(self.full_detail_volume_node)
            else:
                self.full_detail_volume_node = slicer.util.addVolumeFromArray(data)
                self.full_detail_volume_node.name = f"Full detail sinogram"

            self.full_detail_volume_node.GetDisplayNode().SetAutoWindowLevel(False)
            self.full_detail_volume_node.GetDisplayNode().SetWindowLevelMinMax(self.sampleData.sinogram_min, self.sampleData.sinogram_max)
            
            # Update outline bounds.
            bounds = np.empty(6)
            self.full_detail_volume_node.GetBounds(bounds)
            self.sceneObjects.sinogramOutline.SetBounds(*bounds)

            end = time.time()
            print(f"[DEBUG] Adding/modifying volume took: {end - start} s")

            if self.full_detail_volume_node:
                for view in ["Red", "Green", "Yellow"]:
                    slicer.app.layoutManager().sliceWidget(view).sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(self.full_detail_volume_node.GetID())
                slicer.app.layoutManager().resetSliceViews()
            else:
                slicer.util.errorDisplay("Failed to load sinogram slice.")

        except Exception as e:
            import traceback
            print(traceback.format_exc())

    def playButtonToggled(self, checked):
        if checked:
            self.playButtonTimer.start()
            self.ui.playButton.text = "Stop"
        else:
            self.playButtonTimer.stop()
            self.ui.playButton.text = "Play"

    def advanceSinogramSlice(self):
        self.ui.indexSlider.value += int(self.ui.playSpeedSlider.value)

# -----------------------------
# HTTP helpers (use saved base URL)
# -----------------------------
def stream_nrrd_from_url(filename, base_url: Optional[str] = None):
    base = normalize_base_url(base_url or get_saved_base_url())
    url = f"{base}/images/{filename}"
    temp_file_path = os.path.join(slicer.app.temporaryPath, filename)
    
    try:
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

class SinoReconsVisual2Logic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)


class SinoReconsVisual2Test(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.delayDisplay("Test not implemented.")