import os
import json
import logging
import requests
import qt
import slicer
import vtk
import time
import io
import uuid
import traceback
# This is needed for avif support in pillow. 
# 3D slicer needs to be restarted for the upgrade to properly load.
#slicer.util.pip_install("--upgrade pillow")
import PIL
import PIL.Image
import numpy as np
import typing
import re
from pathlib import Path
from typing import Optional
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy, numpy_to_vtkIdTypeArray
from dataclasses import dataclass

import slicer.packaging
slicer.packaging.pip_ensure("pynrrd", requester="SinoReconsVisual2")
import nrrd

slicer.packaging.pip_ensure("pathvalidate", requester="SinoReconsVisual2")
import pathvalidate

from urllib.parse import urlparse
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

class VersionNotSupportedError(Exception):
    def __init__(self, message, version) -> None:
        super().__init__(message)
        self.version = version

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

    roiSinogramRangeModel: slicer.vtkMRMLModelNode
    roiSinogramRangeModelDisplay: slicer.vtkMRMLModelDisplayNode

    roiSinogramTrajectoryModel: slicer.vtkMRMLModelNode
    roiSinogramTrajectoryModelDisplay: slicer.vtkMRMLModelNode

    def __init__(self):
        pass

class SampleData:
    specie: str
    tree_ID: int
    disk_ID: int

    totalSamples: int

    sinogram_min: np.float32
    sinogram_max: np.float32

    bounds_min: np.typing.NDArray[np.float32]
    bounds_max: np.typing.NDArray[np.float32]

    geometry: dict[str, np.typing.NDArray]

    def __init__(self):
        pass

@dataclass
class ROIJsonData:
    uuid: uuid.UUID
    position: str
    size: str
    sinogram_start_index: int
    sinogram_end_index: int

class ROIData:
    name: str
    uuid: uuid.UUID
    roi_node: slicer.vtkMRMLMarkupsROINode
    roi_list_widget: qt.QListWidgetItem
    sinogram_start_index: int
    sinogram_end_index: int
    _modified: bool = False

    original_path: Path|None = None

    def to_data(self) -> ROIJsonData:
        center = self.roi_node.GetCenter()
        size = self.roi_node.GetSize()
        center_str = f"{center.GetX()},{center.GetY()},{center.GetZ()}"
        size_str = f"{size[0]},{size[1]},{size[2]}"
        return ROIJsonData(self.uuid, center_str, size_str, self.sinogram_start_index, self.sinogram_end_index)

class QROINameValidator(qt.QValidator):
    roi_list: qt.QListWidget

    def __init__(self, parent = None):
        super().__init__(parent)

    def validate(self, input: str, pos: int) -> qt.QValidator.State:
        #print(f"validate: {input}")
        try:
            pathvalidate.validate_filename(f"{input}.json", platform=pathvalidate.Platform.UNIVERSAL)
        except pathvalidate.ValidationError as e:
            print(f"{e}")
            if e.reason == pathvalidate.ErrorReason.RESERVED_NAME:
                print(f"validate: {input} reserved")
                return qt.QValidator.Intermediate
            else:
                print(f"validate: {input} invalid")
                return qt.QValidator.Invalid
        print(f"validate: {input} ok")
        return qt.QValidator.Acceptable
            

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
        self.playButtonTimer = qt.QTimer()
        self.playButtonTimer.setInterval(10)
        self.playButtonTimer.timeout.connect(self.advanceSinogramSlice)

        # Slider debounce timer
        self.sliderDebounceTimer = qt.QTimer()
        self.sliderDebounceTimer.setSingleShot(True)
        self.sliderDebounceTimer.setInterval(1)  # milliseconds
        self.sliderDebounceTimer.timeout.connect(self.loadPreviewSlice)

        # Timer to debounce loading full detail (float32) slice
        self.fullDetailDebounceTimer = qt.QTimer()
        self.fullDetailDebounceTimer.setSingleShot(True)
        self.fullDetailDebounceTimer.setInterval(500)
        self.fullDetailDebounceTimer.timeout.connect(self.loadFullDetailSlice)

        # FIXME: For some reason it seems like we can't do this here
        # because slicer.mrmlScene isn't set here or something similar.
        # - Julius Häger 2026-03-05
        self.sceneObjects = self.createSceneObjects()

        print(self.ui.roiCenterCoordinateWidget.coordinatesChanged.typeName())
        print(self.ui.roiCenterCoordinateWidget.coordinatesChanged.parameterTypes())
        print(self.ui.roiCenterCoordinateWidget.coordinatesChanged.parameterNames())
        #self.ui.roiCenterCoordinateWidget.coordinatesChanged.connect(self.roiCenterChanged)
        #self.ui.roiSizeCoordinateWidget.coordinatesChanged.connect(self.roiSizeChanged)
        self.ui.roiCenterCoordinateWidget.connect("coordinatesChanged(double, double, double, double)", self.roiCenterChanged)
        self.ui.roiSizeCoordinateWidget.connect("coordinatesChanged(double, double, double, double)", self.roiSizeChanged)
        self.ui.sinogramRangeWidget.valuesChanged.connect(self.roiSinogramValuesChanged)

        self.ui.addROIButton.clicked.connect(self.addNewROIClicked)
        self.ui.removeROIButton.clicked.connect(self.removeROIClicked)
        self.ui.roiListWidget.currentItemChanged.connect(self.selectROI)
        self.ui.roiListWidget.itemChanged.connect(self.roiItemChanged)
        self.selectROI(None, None)
        self.ui.roiListWidgetEventFilter = self.ROIListEventFilter(self.ui.roiListWidget)
        self.ui.roiSaveButton.clicked.connect(self.saveROIClicked)

    class ROIListEventFilter(qt.QObject):
        roi_list_widget: qt.QListWidget
        roi_list_widget_viewport: typing.Any # FIXME: real type..

        def __init__(self, roi_list_widget: qt.QListWidget):
            super().__init__()
            self.roi_list_widget = roi_list_widget
            self.roi_list_widget.installEventFilter(self)

            self.roi_list_widget_viewport = self.roi_list_widget.viewport()
            self.roi_list_widget_viewport.installEventFilter(self)
        
        def eventFilter(self, source, event) -> bool:
            # FIXME: InputMethodQueryEvents cause all kinds of havock crashing 3D Slicer all together.
            # The crashing only happens sometimes and it seems to be related to recursive calls to eventFilter
            # though I have no idea how this function is getting called recursively...
            # - Julius Häger 2026-04-08
            #if event.type() == qt.QEvent.InputMethodQuery:
            #    return False
            
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
            itemData: ROIData = item.data(qt.Qt.UserRole)
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
            self.sampleData.specie = str(sample["specie"])
            self.sampleData.tree_ID = int(sample["tree_ID"])
            self.sampleData.disk_ID = int(sample["disk_ID"])
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
            self.onIndexChanged(0)
            self.setImageOnSensor(self.ui.showSinogramOnSensorCheckbox.checkState())

            self.ui.sinogramWidget.setEnabled(True)
            self.ui.reconstructionWidget.setEnabled(True)
            self.ui.roiWidget.setEnabled(True)
            self.ui.sinogramRangeWidget.setRange(0, self.sampleData.totalSamples - 1)

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

            sample_roi_dir = Path(os.path.expanduser(f"~/Documents/SinoRecons/{self.sampleData.specie}_{self.sampleData.tree_ID}_{self.sampleData.disk_ID}/"))
            self.clearROIs()
            self.loadROIsForSample(sample_roi_dir)

        except Exception as e:
            print(traceback.format_exc())
            slicer.util.errorDisplay(f"Failed to load full dataset:\n{e}")

    class QListWidgetItemModifiedDelegate(qt.QStyledItemDelegate):
        def __init__(self, parent):
            super().__init__(parent)

        def listContainsName(self, name: str, ignore_index: int) -> bool:
            list = self.parent()
            for row in range(list.count):
                if row == ignore_index:
                    continue
                item = list.item(row)
                print(item.text())
                if name == item.text():
                    return True
            return False

        def createEditor(self, parent, option, index) -> qt.QWidget:
            widget = qt.QStyledItemDelegate.createEditor(self, parent, option, index)
            widget.setValidator(QROINameValidator(widget))
            return widget

        def setModelData(self, editor: qt.QWidget, model: qt.QAbstractItemModel, index: qt.QModelIndex) -> None:
            roi_data: ROIData = index.data(qt.Qt.UserRole)
            user_property = editor.metaObject().userProperty().name()
            edit_str: str = editor.property(user_property)

            if self.listContainsName(edit_str, index.row()):
                match = re.match("\\((\\d+)\\)$", edit_str)
                if match:
                    edit_str = edit_str.removesuffix(f"({match.group(1)})") + f"({int(match.group(1)) + 1})"
                else:
                    edit_str = edit_str + " (1)"

            if edit_str != roi_data.name:
                roi_data.name = edit_str
                # FIXME: This is a hack as we can't call roiUpdateModified here.
                # Instead we mark it as modified here and then in roiItemChanged
                # we call roiUpdateModified with the updated _modified value.
                # - Julius Häger 2026-04-29
                roi_data._modified = True
            if (roi_data._modified):
                edit_str += "*"
            model.setData(index, edit_str, qt.Qt.EditRole)
            #with open("test.txt", "a") as f:
            #    f.write(f"setModelData {editor} {model} {index} editor user property: {editor.metaObject().userProperty().name()}, index EditRole data: {index.data(qt.Qt.EditRole)} UserRole {index.data(qt.Qt.UserRole).name}\n")

        def setEditorData(self, editor: qt.QWidget, index: qt.QModelIndex) -> None:
            #print(f"setEditorData {editor} {index} {editor.data(qt.Qt.UserRole)}")
            roi_data: ROIData = index.data(qt.Qt.UserRole)
            user_property = editor.metaObject().userProperty().name()
            editor.setProperty(user_property, roi_data.name)
            #with open("test.txt", "a") as f:
            #    f.write(f"setEditorData {editor} {index} {editor.metaObject().userProperty().name()} index EditRole data: {index.data(qt.Qt.EditRole)} UserRole {index.data(qt.Qt.UserRole).name}\n")

    def addNewROIClicked(self):
        self.addROI()

    def addROI(self,
               name: str|None = None,
               id: uuid.UUID|None = None,
               center: vtk.vtkVector3f|None = None,
               size: vtk.vtkVector3f|None = None,
               sinogram_start: int|None = None,
               sinogram_end: int|None = None,
               file: Path|None = None) -> ROIData:
        
        def roiListWidgetHasName(name: str) -> bool:
            for row in range(self.ui.roiListWidget.count):
                item = self.ui.roiListWidget.item(row)
                if name == item.text():
                    return True
            return False

        if name is None:
            counter = 1
            name = f"roi {self.ui.roiListWidget.count + counter}"
            while roiListWidgetHasName(name):
                counter += 1
                name = f"roi {self.ui.roiListWidget.count + counter}"
        
        roi_data = ROIData()
        roi_data.name = name
        roi_data.uuid = uuid.uuid4() if id is None else id # uuid4 = Random uuid
        roi_data.roi_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        roi_data.roi_node.GetMarkupsDisplayNode().Visibility2DOff()
        roi_data.roi_node.SetName(name)

        if center is not None:
            roi_data.roi_node.SetCenter(center)
        
        if size is not None:
            roi_data.roi_node.SetSize((size.GetX(), size.GetY(), size.GetZ()))

        # Update UI with the new roi
        roi_data.roi_node.AddObserver(vtk.vtkCommand.ModifiedEvent, self.roiNodeChanged)

        roi_data.sinogram_start_index = 0 if sinogram_start is None else sinogram_start
        roi_data.sinogram_end_index = self.sampleData.totalSamples - 1 if sinogram_end is None else sinogram_end

        roi_data.roi_list_widget = qt.QListWidgetItem()
        roi_data.roi_list_widget.setText(name)
        roi_data.roi_list_widget.setFlags(qt.Qt.ItemIsEditable | qt.Qt.ItemIsEnabled)
        roi_data.roi_list_widget.setData(qt.Qt.UserRole, roi_data)

        self.setInteractive(roi_data.roi_list_widget, False)

        self.ui.roiListWidget.addItem(roi_data.roi_list_widget)
        self.ui.roiListWidget.setCurrentItem(roi_data.roi_list_widget)
        self.ui.roiListWidget.setItemDelegate(self.QListWidgetItemModifiedDelegate(self.ui.roiListWidget))

        roi_data.original_path = file

        self.roiUpdateModified(roi_data, True)

        return roi_data

    def roiItemChanged(self, item: qt.QListWidgetItem):
        itemData: ROIData = item.data(qt.Qt.UserRole)
        new_name = item.text()
        print(f"roiItemChanged: {new_name} old name: {itemData.name}")
        itemData.roi_node.SetName(new_name)
        # This is the second part of the hack in QListWidgetItemModifiedDelegate.
        # At this point the delegate has updated _modified after the list item was edited
        # so now we must actually apply the modified update.
        # - Julius Häger 2026-04-29
        self.roiUpdateModified(itemData, itemData._modified)

    def clearROIs(self):
        for row in reversed(range(self.ui.roiListWidget.count)):
            item = self.ui.roiListWidget.takeItem(row)
            itemData: ROIData = item.data(qt.Qt.UserRole)
            slicer.mrmlScene.RemoveNode(itemData.roi_node)

    def removeROIClicked(self):
        if self.ui.roiListWidget.currentRow != -1:
            item = self.ui.roiListWidget.takeItem(self.ui.roiListWidget.currentRow)
            itemData: ROIData = item.data(qt.Qt.UserRole)
            slicer.mrmlScene.RemoveNode(itemData.roi_node)

            if itemData.original_path != None:
                slicer.packaging.pip_ensure("Send2Trash", requester="SinoReconsVisual2")
                import send2trash
                send2trash.send2trash(itemData.original_path)
        
    def selectROI(self, current: qt.QListWidgetItem, previous: qt.QListWidgetItem):
        if current == None:
            self.ui.roiEditWidget.setEnabled(False)
            self.ui.removeROIButton.setEnabled(False)
            self.updateRoiSinogramRange(0, 0, False)
        else:
            self.ui.roiEditWidget.setEnabled(True)
            self.ui.removeROIButton.setEnabled(True)

            itemData: ROIData = current.data(qt.Qt.UserRole)
            roiNode = itemData.roi_node

            center = roiNode.GetCenter()
            size = roiNode.GetSize()

            self.ui.roiCenterCoordinateWidget.blockSignals(True)
            self.ui.roiCenterCoordinateWidget.setCoordinates(center.GetX(), center.GetY(), center.GetZ(), 0)
            self.ui.roiCenterCoordinateWidget.blockSignals(False)

            self.ui.roiSizeCoordinateWidget.blockSignals(True)
            self.ui.roiSizeCoordinateWidget.setCoordinates(size[0], size[1], size[2], 0)
            self.ui.roiSizeCoordinateWidget.blockSignals(False)

            self.ui.sinogramRangeWidget.blockSignals(True)
            self.ui.sinogramRangeWidget.setValues(itemData.sinogram_start_index, itemData.sinogram_end_index)
            self.ui.sinogramRangeWidget.blockSignals(False)
            self.updateRoiSinogramRange(itemData.sinogram_start_index, itemData.sinogram_end_index, True)

            self.ui.roiSaveButton.setEnabled(itemData._modified)
            
        if current != None:
            self.setInteractive(current, True)

        if previous != None:
            self.setInteractive(previous, False)

    def setInteractive(self, listWidgetItem: qt.QListWidgetItem, enable: bool):
        itemData: ROIData = listWidgetItem.data(qt.Qt.UserRole)
        roiNode = itemData.roi_node
        roiDisplayNode = roiNode.GetMarkupsDisplayNode()
        if enable:
            roiDisplayNode.SetSelectedColor(1.0, 0.0, 0.0)
            roiDisplayNode.SetHandlesInteractive(True)
            roiDisplayNode.SetFillOpacity(0.7)
            roiDisplayNode.SetOutlineOpacity(1.0)
            roiDisplayNode.SetTranslationHandleVisibility(True)
            
            roiDisplayNode.SetScaleHandleVisibility(True)
        else:
            roiDisplayNode.SetSelectedColor(0.1, 0.1, 0.1)
            roiDisplayNode.SetHandlesInteractive(False)
            roiDisplayNode.SetFillOpacity(0.2)
            roiDisplayNode.SetOutlineOpacity(0.4)

    def saveROIClicked(self):
        if self.ui.roiListWidget.currentRow != -1:
            item = self.ui.roiListWidget.item(self.ui.roiListWidget.currentRow)
            itemData: ROIData = item.data(qt.Qt.UserRole)
            
            sample_dir = Path(os.path.expanduser(f"~/Documents/SinoRecons/{self.sampleData.specie}_{self.sampleData.tree_ID}_{self.sampleData.disk_ID}"))
            self.saveROIToFile(itemData, sample_dir)
        else:
            print("[ERROR] There is no ROI selected, can't save.")

    def saveROIToFile(self, roi: ROIData, sample_dir: Path) -> None:
        path =  sample_dir / f"{roi.name}.json"
        # FIXME: Check that the original and new path are in the same directory??
        if path != roi.original_path and roi.original_path != None:
            file_uuid: uuid.UUID
            with open(roi.original_path, "r") as f:
                data = json.load(f)
                file_uuid = uuid.UUID(data["uuid"])
            if file_uuid == roi.uuid:
                os.remove(roi.original_path)
                print(f"Removed old file {roi.original_path} -> {path}")

        data: dict[str, typing.Any] = {}
        data["version"] = 0
        data["uuid"] = str(roi.uuid)
        # FIXME: type
        center: vtk.vtkVector3d = roi.roi_node.GetCenter()
        size = roi.roi_node.GetSize()
        data["center"] = (center.GetX(), center.GetY(), center.GetZ())
        data["size"] = (size[0], size[1], size[2])
        data["sinogram_range"] = (roi.sinogram_start_index, roi.sinogram_end_index)

        # FIXME: If this ROI has changed name we want to delete the old file...
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)
        roi.original_path = path
        self.roiUpdateModified(roi, False)
        print(f"Saved roi to {path.absolute()}")

    def loadROIsForSample(self, sample_directory: Path) -> None:
        if not Path.exists(sample_directory):
            return
        paths = os.listdir(sample_directory)
        for path in paths:
            print(f"Loading {path} {os.path.isfile(sample_directory / path)} {path.endswith(".json")}")
            if os.path.isfile(sample_directory / path) and path.endswith(".json"):
                print(f"Loading {path}")
                self.loadROIFromFile(sample_directory / path)

    def loadROIFromFile(self, path: Path):
        data: dict[str, typing.Any]
        with open(path, "r") as f:
            data = json.load(f)
        version = int(data["version"])
        if version != 0:
            raise VersionNotSupportedError("The ROI file version is newer than this plugin understands. Maybe there is a new plugin version?", version)
        name = path.stem
        id = uuid.UUID(data["uuid"])
        center = vtk.vtkVector3f(data["center"])
        size = vtk.vtkVector3f(data["size"])
        sinogram_start_index = int(data["sinogram_range"][0])
        sinogram_end_index = int(data["sinogram_range"][1])

        file_mtime = os.path.getmtime(path)

        # FIXME: Check if the uuid is unique
        should_add: bool = True
        for row in reversed(range(self.ui.roiListWidget.count)):
            item = self.ui.roiListWidget.item(row)
            itemData: ROIData = item.data(qt.Qt.UserRole)
            if itemData.uuid == id:
                if itemData.original_path == None:
                    # We are the source of truth, remove this item and keep this item
                    should_add = True
                    
                    # Remove the existing item from the list
                    item = self.ui.roiListWidget.takeItem(row)
                    itemData: ROIData = item.data(qt.Qt.UserRole)
                    slicer.mrmlScene.RemoveNode(itemData.roi_node)

                    print(f"Duplicate uuid found. But existing item did not have a file, so we use '{path}'.")
                else:
                    mtime = os.path.getmtime(itemData.original_path)
                    if file_mtime > mtime:
                        # The file we are reading right now is the source of truth...
                        should_add = True

                        # Remove the existing item from the list
                        item = self.ui.roiListWidget.takeItem(row)
                        itemData: ROIData = item.data(qt.Qt.UserRole)
                        slicer.mrmlScene.RemoveNode(itemData.roi_node)

                        print(f"Duplicate uuid found for files '{itemData.original_path}' and '{path}'. '{path}' had newer modification date so I'm using that.")
                    else:
                        # The already loaded file is the source of truth. 
                        # Skip loading this file.
                        should_add = False
                        print(f"Duplicate uuid found for files '{itemData.original_path}' and '{path}'. '{itemData.original_path}' had newer modification date so I'm using that.")

        if should_add:
            roi = self.addROI(name, id, center, size, sinogram_start_index, sinogram_end_index, path)
            self.roiUpdateModified(roi, False)

    # PythonQt seems to bind the parameter as a 'double'
    # instead of a 'double*' meaning the parameter is
    # completely useless in python.
    # - Julius Häger 2026-04-07
    # https://github.com/commontk/CTK/pull/1417 updates this and adds a x,y,z,w overload
    # - Julius Häger 2026-04-29
    def roiCenterChanged(self, x: float, y: float, z: float, w: float):
        if self.ui.roiListWidget.currentRow == -1:
            return
        
        item = self.ui.roiListWidget.item(self.ui.roiListWidget.currentRow)
        itemData: ROIData = item.data(qt.Qt.UserRole)
        roiNode = itemData.roi_node

        roiNode.SetCenter([x, y, z])
        self.roiUpdateModified(itemData, True)

    def roiSizeChanged(self, x: float, y: float, z: float, w: float):
        if self.ui.roiListWidget.currentRow == -1:
            return
        
        item = self.ui.roiListWidget.item(self.ui.roiListWidget.currentRow)
        itemData: ROIData = item.data(qt.Qt.UserRole)
        roiNode = itemData.roi_node

        roiNode.SetSize([x, y, z])

        self.roiUpdateModified(itemData, True)

    def getROIDataFromNode(self, roi_node: slicer.vtkMRMLMarkupsROINode) -> ROIData|None:
        for row in range(self.ui.roiListWidget.count):
            item = self.ui.roiListWidget.item(row)
            itemData: ROIData = item.data(qt.Qt.UserRole)
            if itemData.roi_node == roi_node:
                return itemData
        return None
    
    # Called when the roi is changed using the 3D/2D widgets.
    def roiNodeChanged(self, roiNode, event):
        # FIXME: Only change the UI if this is the active roiNode?
        center = roiNode.GetCenter()
        size = roiNode.GetSize()

        # FIXME: Check that anything actually changed?
        modified = False

        # FIXME: Very inefficient to set the coordinates through strings
        # FIXME: The coordinate string that is returned is rounded to the number of decimals used to display
        # which means we will never be able to accurately get the values from the control... so the modified check will fail.
        self.ui.roiCenterCoordinateWidget.blockSignals(True)
        old_center = [self.ui.roiCenterCoordinateWidget.getCoordinate(0), self.ui.roiCenterCoordinateWidget.getCoordinate(1), self.ui.roiCenterCoordinateWidget.getCoordinate(2)]
        if old_center != [center.GetX(),center.GetY(),center.GetZ()]:
            print(f"change center! {old_center} -> {[center.GetX(),center.GetY(),center.GetZ()]}")
            #self.ui.roiCenterCoordinateWidget.coordinates = new_center_str
            self.ui.roiCenterCoordinateWidget.setCoordinates(center.GetX(), center.GetY(), center.GetZ(), 0)
            modified = True
        self.ui.roiCenterCoordinateWidget.blockSignals(False)


        self.ui.roiSizeCoordinateWidget.blockSignals(True)
        old_size = [self.ui.roiSizeCoordinateWidget.getCoordinate(0), self.ui.roiSizeCoordinateWidget.getCoordinate(1), self.ui.roiSizeCoordinateWidget.getCoordinate(2)]
        if old_size != [size[0],size[1],size[2]]:
            print(f"change size! {old_size} -> {[size[0],size[1],size[2]]}")
            self.ui.roiSizeCoordinateWidget.setCoordinates(size[0],size[1],size[2], 0)
            modified = True
        self.ui.roiSizeCoordinateWidget.blockSignals(False)

        roi = self.getROIDataFromNode(roiNode)
        if roi is not None and modified:
            self.roiUpdateModified(roi, True)

    def roiSinogramValuesChanged(self, minVal: float, maxVal: float):
        if self.ui.roiListWidget.currentRow == -1:
            return

        minVal = int(minVal)
        maxVal = int(maxVal)

        item = self.ui.roiListWidget.item(self.ui.roiListWidget.currentRow)
        itemData: ROIData = item.data(qt.Qt.UserRole)
        itemData.sinogram_start_index = minVal
        itemData.sinogram_end_index = maxVal

        self.updateRoiSinogramRange(minVal, maxVal, True)

        self.roiUpdateModified(itemData, True)

    def roiUpdateModified(self, roi: ROIData, modified: bool):
        roi._modified = modified

        #import inspect
        if roi._modified:
            #print(f"{roi.name} is modified {inspect.stack()[1][3]} {inspect.stack()[2][3]} {inspect.stack()[3][3]}")
            roi.roi_list_widget.setText(f"{roi.name}*")
        else:
            #print(f"{roi.name} no longer modified {inspect.stack()[1][3]} {inspect.stack()[2][3]} {inspect.stack()[3][3]}")
            roi.roi_list_widget.setText(roi.name)

        self.ui.roiSaveButton.setEnabled(roi._modified)

    # -----------------------------
    # Rendering helpers
    # -----------------------------
    def createSceneObjects(self):
        sceneObjects = Vtk3DSceneObjects()
        #print(slicer.mrmlScene)

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
        sceneObjects.sensorModelDisplay.SetRepresentation(slicer.vtkMRMLDisplayNode.SurfaceRepresentation)
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
        sceneObjects.sinogramOutlineDisplay.VisibilityOff()

        sceneObjects.roiSinogramRangeModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "ROI Sinogram Range")
        polys = vtk.vtkPolyData()
        polys.SetPoints(vtk.vtkPoints())
        polys.SetVerts(vtk.vtkCellArray())
        sceneObjects.roiSinogramRangeModel.SetAndObservePolyData(polys)
        sceneObjects.roiSinogramRangeModelDisplay = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        sceneObjects.roiSinogramRangeModel.SetAndObserveDisplayNodeID(sceneObjects.roiSinogramRangeModelDisplay.GetID())
        sceneObjects.roiSinogramRangeModelDisplay.SetColor((0.5765, 0.8431, 0.8118))
        sceneObjects.roiSinogramRangeModelDisplay.SetPointSize(8)
        sceneObjects.roiSinogramRangeModelDisplay.SetVisibility(1)

        sceneObjects.roiSinogramTrajectoryModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "ROI Sinogram Trajectory")
        polys = vtk.vtkPolyData()
        polys.SetPoints(vtk.vtkPoints())
        polys.SetLines(vtk.vtkCellArray())
        sceneObjects.roiSinogramTrajectoryModel.SetAndObservePolyData(polys)
        sceneObjects.roiSinogramTrajectoryModelDisplay = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        sceneObjects.roiSinogramTrajectoryModel.SetAndObserveDisplayNodeID(sceneObjects.roiSinogramTrajectoryModelDisplay.GetID())
        sceneObjects.roiSinogramTrajectoryModelDisplay.SetColor((0.5765, 0.8431, 0.8118))
        sceneObjects.roiSinogramTrajectoryModelDisplay.SetLineWidth(4)
        sceneObjects.roiSinogramTrajectoryModelDisplay.SetVisibility(1)

        #93d7cf

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
            slicer.mrmlScene.RemoveNode(self.sceneObjects.roiSinogramRangeModelDisplay)
            slicer.mrmlScene.RemoveNode(self.sceneObjects.roiSinogramRangeModel)
            slicer.mrmlScene.RemoveNode(self.sceneObjects.roiSinogramTrajectoryModelDisplay)
            slicer.mrmlScene.RemoveNode(self.sceneObjects.roiSinogramTrajectoryModel)
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
            self.sceneObjects.roiSinogramRangeModelDisplay = None
            self.sceneObjects.roiSinogramRangeModel = None
            self.sceneObjects.roiSinogramTrajectoryModelDisplay = None
            self.sceneObjects.roiSinogramTrajectoryModel = None

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

    def updateRoiSinogramRange(self, start_index: int, end_index: int, is_visible: bool):
        if is_visible:
            sources = self.sampleData.geometry.get("sources", np.empty(0))
            start_source = sources[start_index].reshape(1, 3)
            end_source = sources[end_index].reshape(1, 3)
            start_and_end = np.vstack((start_source, end_source))

            polyData : vtk.vtkPolyData = typing.cast(vtk.vtkPolyData, self.sceneObjects.roiSinogramRangeModel.GetPolyData())

            points = typing.cast(vtk.vtkPoints, polyData.GetPoints())
            points.SetData(numpy_to_vtk(start_and_end, 1))

            cells = polyData.GetVerts()
            idx = numpy_to_vtkIdTypeArray(np.arange(len(start_and_end)), 1)
            cells.SetData(len(start_and_end), idx)

            points.Modified()
            cells.Modified()
            polyData.Modified()

            trajectory = self.sampleData.geometry.get("full_trajectory", np.empty(0))
            trajectory_range = trajectory[start_index:end_index]

            polyData : vtk.vtkPolyData = typing.cast(vtk.vtkPolyData, self.sceneObjects.roiSinogramTrajectoryModel.GetPolyData())

            points = typing.cast(vtk.vtkPoints, polyData.GetPoints())
            points.SetData(numpy_to_vtk(trajectory_range, 1))

            cells = polyData.GetLines()
            idx = numpy_to_vtkIdTypeArray(np.arange(len(trajectory_range)), 1)
            cells.SetData(len(trajectory_range), idx)

            points.Modified()
            cells.Modified()
            polyData.Modified()

        self.sceneObjects.roiSinogramRangeModelDisplay.SetVisibility(1 if is_visible else 0)
        self.sceneObjects.roiSinogramTrajectoryModelDisplay.SetVisibility(1 if is_visible else 0)


    def setImageOnSensor(self, state : int):
        if hasattr(self, "sceneObjects") == False or self.sceneObjects == None:
            return
        
        print(f"show on sensor: {state}")
        if state == 2:
            self.sceneObjects.sensorModelDisplay.SetTextureImageDataConnection(self.sceneObjects.sensorModelImageProducer.GetOutputPort())
            self.sceneObjects.sensorModelDisplay.SetOpacity(1.0)
            self.sceneObjects.sensorModelDisplay.SetInterpolation(slicer.vtkMRMLDisplayNode.FlatInterpolation)
            self.sceneObjects.sensorModelDisplay.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseDataScalarRange) # Data range = auto
        else:
            self.sceneObjects.sensorModelDisplay.SetTextureImageDataConnection(None)
            self.sceneObjects.sensorModelDisplay.SetOpacity(0.7)
            self.sceneObjects.sensorModelDisplay.SetInterpolation(slicer.vtkMRMLDisplayNode.PhongInterpolation)

    # -----------------------------
    # Slider / sinogram fetch
    # -----------------------------
    def onIndexChanged(self, value):
        start = time.time()
        self.currentIndex = value
        self.ui.sliderIndexLabel.setText(f"Index: {value}")
        self.updateSceneData(value)
        if self.playButtonTimer.isActive():
            self.loadPreviewSlice()
        else:
            self.sliderDebounceTimer.start()
        self.fullDetailDebounceTimer.start()
        end = time.time()
        print(f"[DEBUG] onIndexChanged took {end - start} seconds")

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
            self.sceneObjects.sinogramOutlineDisplay.VisibilityOn()

            end = time.time()
            print(f"[DEBUG] Adding/modifying volume took: {end - start} s")

            start = time.time()
            tex_data = (np.iinfo(np.uint8).max * (img_data_mapped - self.sampleData.sinogram_min) / (self.sampleData.sinogram_max - self.sampleData.sinogram_min)).astype(np.uint8)
            # FIXME: Do we need to flip here? What is the correct way to show this?
            # Alternatively the UV coordinates on the sensor geometry is "wrong"...
            tex_data = np.flip(tex_data, axis=1)

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

            #im.save(f"C:/Users/juliu/Documents/SlicerCapture/{index}_q100_s0_400.avif", format="", max_threads=1, quality=100, speed=0, subsampling="4:0:0")
            #im.save(f"C:/Users/juliu/Documents/SlicerCapture/{index}_q57_s7_400.avif", format="", max_threads=1, quality=57, speed=7, subsampling="4:0:0")
            #im.save(f"C:/Users/juliu/Documents/SlicerCapture/{index}_q54_s5_400.avif", format="", max_threads=1, quality=54, speed=5, subsampling="4:0:0")
            #im.save(f"C:/Users/juliu/Documents/SlicerCapture/{index}_q52_s5_400.avif", format="", max_threads=1, quality=52, speed=5, subsampling="4:0:0")
            #im.save(f"C:/Users/juliu/Documents/SlicerCapture/{index}_q48_s5_400.avif", format="", max_threads=1, quality=48, speed=5, subsampling="4:0:0")
            #im.save(f"C:/Users/juliu/Documents/SlicerCapture/{index}_q45_s5_400.avif", format="", max_threads=1, quality=45, speed=5, subsampling="4:0:0")

            #im.save(f"C:/Users/juliu/Documents/SlicerCapture/{index}_q57_s8_400.avif", format="", max_threads=1, quality=57, speed=8, subsampling="4:0:0")
            #im.save(f"C:/Users/juliu/Documents/SlicerCapture/{index}_q57_s6_400.avif", format="", max_threads=1, quality=57, speed=6, subsampling="4:0:0")
            #im.save(f"C:/Users/juliu/Documents/SlicerCapture/{index}_q57_s4_400.avif", format="", max_threads=1, quality=57, speed=4, subsampling="4:0:0")
            #im.save(f"C:/Users/juliu/Documents/SlicerCapture/{index}_q57_s2_400.avif", format="", max_threads=1, quality=57, speed=2, subsampling="4:0:0")
            #im.save(f"C:/Users/juliu/Documents/SlicerCapture/{index}_q57_s0_400.avif", format="", max_threads=1, quality=57, speed=0, subsampling="4:0:0")

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
            self.sceneObjects.sinogramOutlineDisplay.VisibilityOn()

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

        progress_dialog = qt.QProgressDialog("Loading file...", "Cancel", 0, max_progress)
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
    def setup(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setup()
        self.delayDisplay("Test not implemented.")