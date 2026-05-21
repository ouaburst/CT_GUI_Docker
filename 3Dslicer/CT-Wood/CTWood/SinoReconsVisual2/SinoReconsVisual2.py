import os
import sys
import json
import logging
import requests
import qt
import slicer
import slicer.packaging
import vtk
import ctk
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
from collections.abc import Iterable

from urllib.parse import urlparse
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

from settings.SinoReconsVisual2SettingsPanel import SinoReconsVisual2SettingsPanel

class VersionNotSupportedError(Exception):
    def __init__(self, message, version) -> None:
        super().__init__(message)
        self.version = version

# -----------------------------
# Settings helpers
# -----------------------------
SETTINGS_KEY_BACKEND_URL = "SinoReconsVisual2/BackendUrl"
DEFAULT_BACKEND_URL = "http://b9398.research.ltu.se:8000"

SETTINGS_KEY_SHOW_SOURCE_DETECTOR = "SinoReconsVisual2/ShowSourceDetector"
SETTINGS_KEY_SHOW_SINOGRAM_ON_SENSOR = "SinoReconsVisual2/ShowSinogramOnSensor"
SETTINGS_KEY_SHOW_REGIONS_OF_INTEREST = "SinoReconsVisual2/ShowRegionsOfInterest"
SETTINGS_KEY_SHOW_REGIONS_OF_INTEREST_SOURCE_DETECTOR = "SinoReconsVisual2/ShowRegionsOfInterestSinogramRangeSourceDetector"

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
        iconPath = os.path.join(get_icons_folder(), "SinoReconsVisual2_v2.svg")
        if os.path.exists(iconPath):
            self.parent.icon = qt.QIcon(iconPath)

        slicer.packaging.pip_install("pynrrd", requester="SinoReconsVisual2")
        slicer.packaging.pip_install("pathvalidate", requester="SinoReconsVisual2")

class SourceDetectorObjects:
    sourceModel: slicer.vtkMRMLModelNode
    sourceModelDisplay: slicer.vtkMRMLModelDisplayNode

    fovRaysModel: slicer.vtkMRMLModelNode
    fovRaysModelDisplay: slicer.vtkMRMLModelDisplayNode

    sensorModel: slicer.vtkMRMLModelNode
    sensorModelDisplay: slicer.vtkMRMLModelDisplayNode
    sensorModelImage: vtk.vtkImageData
    sensorModelImageProducer: vtk.vtkTrivialProducer

class Vtk3DSceneObjects:
    sourceDetectorObjects: SourceDetectorObjects

    trajectoryModel: slicer.vtkMRMLModelNode
    trajectoryModelDisplay: slicer.vtkMRMLModelDisplayNode

    reconOutline: vtk.vtkOutlineSource
    reconCubeModel: slicer.vtkMRMLModelNode
    reconCubeModelDisplay: slicer.vtkMRMLModelDisplayNode

    sinogramOutline: vtk.vtkOutlineSource
    sinogramOutlineNode: slicer.vtkMRMLModelNode
    sinogramOutlineDisplay: slicer.vtkMRMLModelDisplayNode

    roiSinogramRangeModel: slicer.vtkMRMLModelNode
    roiSinogramRangeModelDisplay: slicer.vtkMRMLModelDisplayNode

    sinogramRangeStartSourceDetector: SourceDetectorObjects
    sinogramRangeEndSourceDetector: SourceDetectorObjects

    roiSinogramTrajectoryModel: slicer.vtkMRMLModelNode
    roiSinogramTrajectoryModelDisplay: slicer.vtkMRMLModelNode

    def __init__(self):
        pass

class SampleData:
    specie: str
    tree_ID: int
    disk_ID: int

    totalSamples: int

    sinogram_shape: tuple[int, int, int]

    sinogram_min_value: np.float32
    sinogram_max_value: np.float32

    bounds_min: np.typing.NDArray[np.float32]
    bounds_max: np.typing.NDArray[np.float32]

    rec_bounds_min: np.typing.NDArray[np.float32]
    rec_bounds_max: np.typing.NDArray[np.float32]

    geometry: dict[str, np.typing.NDArray]

    metadata: dict[str, typing.Any]

    def __init__(self):
        pass

@dataclass
class ROIJsonData:
    uuid: uuid.UUID
    position: str
    size: str
    sinogram_start_index: int
    sinogram_end_index: int
    resolution_x: int
    resolution_y: int
    resolution_z: int
    auto_resolution: bool

class ROIData:
    name: str
    uuid: uuid.UUID
    roi_node: slicer.vtkMRMLMarkupsROINode
    roi_list_widget: qt.QListWidgetItem
    sinogram_start_index: int
    sinogram_end_index: int
    resolution: tuple[int, int, int]
    auto_resolution: bool
    _modified: bool = False

    original_path: Path|None = None

    def to_data(self) -> ROIJsonData:
        center = self.roi_node.GetCenter()
        size = self.roi_node.GetSize()
        center_str = f"{center.GetX()},{center.GetY()},{center.GetZ()}"
        size_str = f"{size[0]},{size[1]},{size[2]}"
        return ROIJsonData(self.uuid, center_str, size_str, self.sinogram_start_index, self.sinogram_end_index, self.resolution[0], self.resolution[1], self.resolution[2], self.auto_resolution)

class QROINameValidator(qt.QValidator):
    roi_list: qt.QListWidget

    def __init__(self, parent = None):
        super().__init__(parent)
        

    def validate(self, input: str, pos: int) -> qt.QValidator.State:
        import pathvalidate
        try:
            pathvalidate.validate_filename(f"{input}.json", platform=pathvalidate.Platform.UNIVERSAL)
        except pathvalidate.ValidationError as e:
            if e.reason == pathvalidate.ErrorReason.RESERVED_NAME:
                return qt.QValidator.Intermediate
            else:
                return qt.QValidator.Invalid
        return qt.QValidator.Acceptable
            

def getFirstChildOfType(widget: qt.QWidget, ofType: type) -> qt.QObject:
    for child in widget.children():
        if type(child) == ofType:
            return child

class SinoReconsVisual2Widget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    sceneObjects: Vtk3DSceneObjects
    sampleData: SampleData

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        qt.QResource.registerResource(self.resourcePath("sinoReconsVisual.rcc"))
        self.sampleData = SampleData()

    # -----------------------------
    # UI & wiring
    # -----------------------------
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        self.session = requests.Session()
        
        # FIXME: I'm unable to figure out how to make resources from qrc files work...
        # - Julius Häger 2026-05-13
        qt.QIcon.setThemeName("light")

        # Hide the default box
        slicer.app.layoutManager().threeDWidget(0).mrmlViewNode().SetBoxVisible(0)
        slicer.app.layoutManager().threeDWidget(0).mrmlViewNode().SetAxisLabelsVisible(0)

        # This is a workaround for ctkSettingsDialog not having a removePanel() function.
        # I've opened a PR to add one here: https://github.com/commontk/CTK/pull/1423
        # If we can't call removePanel() then we just add a button to the module UI
        # so that you can still open the settings UI.
        # - Julius Häger 2026-05-19
        self.settingsUI = SinoReconsVisual2SettingsPanel(slicer.util.loadUI(self.resourcePath("UI/SinoReconsVisual2SettingsPanel.ui")), self)
        print(type(slicer.app.settingsDialog()))
        if hasattr(slicer.app.settingsDialog(), 'removePanel'):
            self.settings = qt.QSettings()
            slicer.app.settingsDialog().addPanel("SinoReconsVisual2", qt.QIcon(os.path.join(get_icons_folder(), "SinoReconsVisual2_v2.svg")), self.settingsUI.settingsPanel)
        else:
            # The "Restore Defaults" button in the settings dialog will reset *all* of the settings in the QSettings object associated with that dialog
            # So for this custom settings dialog we can't reuse the default qt.QSettings() object as that would reset *all* slicer settings.
            # To solve this we create our own settings object for the case where we can't add our settings panel to the default settings dialog.
            # - Julius Häger 2026-05-20
            self.settings = qt.QSettings("SinoReconsVisual2", "SinoReconsVisual2")

            self.settingsDialog = ctk.ctkSettingsDialog()
            self.settingsDialog.resetButton = True
            self.settingsDialog.settings = self.settings
            self.settingsDialog.addPanel("SinoReconsVisual2", qt.QIcon(os.path.join(get_icons_folder(), "SinoReconsVisual2_v2.svg")), self.settingsUI.settingsPanel)

            settingsButton = qt.QPushButton()
            settingsButton.text = "Settings"
            settingsButton.clicked.connect(lambda x: self.settingsDialog.exec())
            self.layout.addWidget(settingsButton)

        # Load and attach UI
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SinoReconsVisual2.ui"))
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        self.layout.addWidget(uiWidget)

        self.registerSinogramLayout()

        self.ui.serverUrlLineEdit.setText(self.settings.value(SETTINGS_KEY_BACKEND_URL, DEFAULT_BACKEND_URL))

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

        self.ui.showSourceDetectorCheckBox.stateChanged.connect(self.showSourceDetectorStateChanged)
        self.ui.showSourceDetectorCheckBox.setChecked(self.settings.value(SETTINGS_KEY_SHOW_SOURCE_DETECTOR, True))

        self.ui.showSinogramOnSensorCheckbox.stateChanged.connect(self.showSinogramOnSensorStateChanged)
        self.ui.showSinogramOnSensorCheckbox.setChecked(self.settings.value(SETTINGS_KEY_SHOW_SINOGRAM_ON_SENSOR, False))

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

        self.ui.showROICheckbox.stateChanged.connect(self.showROIs)
        self.ui.showROICheckbox.setChecked(self.settings.value(SETTINGS_KEY_SHOW_REGIONS_OF_INTEREST, False))

        self.ui.showROISinogramRangeSourceDetectorCheckBox.stateChanged.connect(self.showROISinogramRangeSourceDetectorStateChanged)
        self.ui.showROISinogramRangeSourceDetectorCheckBox.setChecked(self.settings.value(SETTINGS_KEY_SHOW_REGIONS_OF_INTEREST_SOURCE_DETECTOR, True))

        if not self.ui.roiCenterCoordinateWidget.connect("coordinatesChanged(double, double, double, double)", self.roiCenterChanged):
            self.ui.roiCenterCoordinateWidget.coordinatesChanged.connect(self.roiCenterChangedOld)
        if not self.ui.roiSizeCoordinateWidget.connect("coordinatesChanged(double, double, double, double)", self.roiSizeChanged):
            self.ui.roiSizeCoordinateWidget.coordinatesChanged.connect(self.roiSizeChangedOld)
        if not self.ui.roiResolutionCoordinateWidget.connect("coordinatesChanged(double, double, double, double)", self.roiResolutionChanged):
            self.ui.roiResolutionCoordinateWidget.coordinatesChanged.connect(self.roiResolutionChangedOld)
        
        self.ui.sinogramRangeWidget.valuesChanged.connect(self.roiSinogramValuesChanged)
        color = self.settings.value("SinoReconsVisual2/ROI/SinogramRangeColor")
        palette = self.ui.sinogramRangeWidget.palette
        palette.setColor(qt.QPalette.Normal, qt.QPalette.Highlight, color)
        palette.setColor(qt.QPalette.Inactive, qt.QPalette.Highlight, color)
        palette.setColor(qt.QPalette.Disabled, qt.QPalette.Highlight, qt.QColor.fromRgbF(color.redF()*0.55, color.greenF()*0.55, color.blueF()*0.55))
        self.ui.sinogramRangeWidget.palette = palette

        self.ui.addROIButton.clicked.connect(self.addNewROIClicked)
        self.ui.removeROIButton.clicked.connect(self.removeROIClicked)
        self.ui.roiListWidget.currentItemChanged.connect(self.selectROI)
        self.ui.roiListWidget.itemChanged.connect(self.roiItemChanged)
        self.selectROI(None, None)
        self.ui.roiListWidgetEventFilter = self.ROIListEventFilter(self.ui.roiListWidget)
        self.ui.roiSaveButton.clicked.connect(self.saveROIClicked)

        self.ui.roiListWidget.setContextMenuPolicy(qt.Qt.CustomContextMenu)
        self.ui.roiListWidget.customContextMenuRequested.connect(self.showROIItemContextMenu)

        self.ui.reconstructionROIComboBox.currentIndexChanged.connect(self.reconstructionROIChanged)

        threeDView = slicer.app.layoutManager().threeDWidget(0).threeDView()
        viewNode = threeDView.mrmlViewNode()
        modelDM = slicer.app.applicationLogic().GetViewDisplayableManagerByClassName(viewNode, "vtkMRMLModelDisplayableManager")
        interactor = threeDView.interactor()
        
        def onLeftClick(caller, event):
            x, y = caller.GetEventPosition()

            interactor = modelDM.GetInteractor()

            y = interactor.GetSize()[1] - y

            if modelDM.Pick(x, y) and modelDM.GetPickedNodeID():
                nodeID = modelDM.GetPickedNodeID()  # this is the *display* node ID
                displayNode = slicer.mrmlScene.GetNodeByID(nodeID)
                dataNode = displayNode.GetDisplayableNode() if displayNode else None
                ras = modelDM.GetPickedRAS()
                cellID = modelDM.GetPickedCellID()
                print(f"Clicked {dataNode.GetName() if dataNode else '(none)'} at RAS={tuple(ras)} cell={cellID}")

        self.interactor_observer_tag = interactor.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, onLeftClick, 1.0)

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
        # viewgroup=196291749 is a hopefully unique viewgroup number so that
        # that slice node doesn't consider itself in the same coordinate space
        # as the other slice nodes that change coordinates depending on what
        # happens in the 3D view. By placing this slice node in a unique view group
        # we prevent it from randomly changing coordinates etc.
        # See: https://discourse.slicer.org/t/decoupled-custom-slice-widget-in-messagebox/1215/14
        # - Julius Häger 2026-05-19
        layoutDesc = """
        <layout type="vertical">
            <item><view class="vtkMRMLViewNode" singletontag="1">
                <property name="viewlabel" action="default">1</property>
                <property name="viewcolor" action="default">#FFFF00</property>
            </view></item>
            <item><view class="vtkMRMLSliceNode" singletontag="Gray">
                <property name="orientation" action="default">Sagittal</property>
                <property name="viewlabel" action="default">Y</property>
                <property name="viewcolor" action="default">#A0A0A0</property>s
                <property name="viewgroup" action="default">196291749</property>
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
            layoutSwitchAction.setIcon(qt.QIcon(os.path.join(get_icons_folder(), "SinoReconsVisual2_v2.svg")))
            layoutSwitchAction.setToolTip("3D and sinogram view")

    def cleanup(self):
        if hasattr(slicer.app.settingsDialog(), 'removePanel'):
            slicer.app.settingsDialog().removePanel(self.settingsUI.settingsPanel)

        if hasattr(self, 'interactor_observer_tag'):
            interactor = slicer.app.layoutManager().threeDWidget(0).threeDView().interactor()
            interactor.RemoveObserver(self.interactor_observer_tag)

        self.destroySceneObjects()
        if hasattr(self, 'volume_node') and self.volume_node:
            slicer.mrmlScene.RemoveNode(self.volume_node)
        if hasattr(self, 'full_detail_volume_node') and self.full_detail_volume_node:
            slicer.mrmlScene.RemoveNode(self.full_detail_volume_node)
        
        if hasattr(self, 'ui'):
            for row in range(self.ui.roiListWidget.count):
                item = self.ui.roiListWidget.item(row)
                itemData: ROIData = item.data(qt.Qt.UserRole)
                slicer.mrmlScene.RemoveNode(itemData.roi_node)

    def onReload(self):
        import importlib
        import settings.SinoReconsVisual2SettingsPanel
        importlib.reload(settings)
        importlib.reload(settings.SinoReconsVisual2SettingsPanel)
        ScriptedLoadableModuleWidget.onReload(self)

    # -----------------------------
    # Utility
    # -----------------------------
    def _currentBaseUrl(self) -> str:
        """Read base URL from line edit (if present) or settings; normalize and persist."""
        if hasattr(self.ui, "serverUrlLineEdit") and self.ui.serverUrlLineEdit is not None:
            entered = self.ui.serverUrlLineEdit.text
            base = normalize_base_url(entered)
            # Persist any change immediately so other actions use it
            self.settings.setValue(SETTINGS_KEY_BACKEND_URL, base)
            # Keep UI clean/normalized
            if entered != base and self.ui.serverUrlLineEdit.text != base:
                self.ui.serverUrlLineEdit.setText(base)
            return base
        # Fallback to settings
        return normalize_base_url(self.settings.value(SETTINGS_KEY_BACKEND_URL, DEFAULT_BACKEND_URL))

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
        else:
            # adjoint or other single-shot methods: no extra params
            pass

        trajectory = self.sampleData.geometry.get("full_trajectory", np.empty(0))
        if self.ui.reconstructionROIComboBox.currentData != None:
            itemData: ROIData = self.ui.reconstructionROIComboBox.currentData
            center = np.array(itemData.roi_node.GetCenter())
            size = np.array(itemData.roi_node.GetSize())
            
            min = center - (size * 0.5)
            max = center + (size * 0.5)

            params["REC_MIN_X"] = float(min[0])
            params["REC_MIN_Y"] = float(min[1])
            params["REC_MIN_Z"] = float(min[2])
            params["REC_MAX_X"] = float(max[0])
            params["REC_MAX_Y"] = float(max[1])
            params["REC_MAX_Z"] = float(max[2])
            params['SINOGRAM_MIN'] = itemData.sinogram_start_index
            params['SINOGRAM_MAX'] = itemData.sinogram_end_index
            params["REC_NPX_X"] = itemData.resolution[0]
            params["REC_NPX_Y"] = itemData.resolution[1]
            params["REC_NPX_Z"] = itemData.resolution[2]
        else:
            # Invalid entry or we've selected the "Whole" entry.
            metadata = self.sampleData.metadata
            params["REC_MIN_X"] = metadata['REC_MIN_X']
            params["REC_MIN_Y"] = metadata["REC_MIN_Y"]
            params["REC_MIN_Z"] = trajectory[0][2]
            params["REC_MAX_X"] = metadata["REC_MAX_X"]
            params["REC_MAX_Y"] = metadata["REC_MAX_Y"]
            params["REC_MAX_Z"] = trajectory[-1][2]
            params['SINOGRAM_MIN'] = 0
            params['SINOGRAM_MAX'] = self.sampleData.totalSamples
            params["REC_NPX_X"] = metadata['REC_NPX_X']
            params["REC_NPX_Y"] = metadata['REC_NPX_Y']
            params["REC_NPX_Z"] = int((params["REC_MAX_Z"] - params["REC_MIN_Z"]) // metadata['REC_PIC_SIZE'])

        params["use_cache"] = self.ui.useCacheCheckbox.checkState() == qt.Qt.Checked

        print(params)

        payload = {
            "specie": specie,
            "tree_ID": int(tree_ID),
            "disk_ID": int(disk_ID),
            "method": method,
            "parameters": params,
        }

        try:
            if slicer.util.confirmOkCancelDisplay("Start reconstruction?", windowTitle="Start ") == False:
                return
            
            import re, json as _json

            slicer.packaging.pip_ensure("requests_futures", requester="SinoReconsVisual2", skip_in_testing = False)
            from requests_futures.sessions import FuturesSession
            from concurrent.futures import Future
            session = FuturesSession()

            future : Future = typing.cast(Future, session.post(run_url, json=payload, stream=True, timeout=None))
            
            progress = slicer.util.createProgressDialog(value=0, maximum=100, labelText='Running reconstruction...', windowTitle='Running reconstruction...')
            progress.labelText = "Waiting for reconstruction"
            progress.value = 99
            while not future.done() and not progress.wasCanceled:
                # FIXME: For iterative method we should actually stream the log and get progress reports that way.
                slicer.app.processEvents()
             
            if progress.wasCanceled:
                # FIXME: Cancel the server operation.
                progress.close()
                return

            r = future.result()
            r.raise_for_status()
            
            # FIXME: The streaming response might be broken as we are now using a future for this?
            # I'm unable to test atm due to an exception being raised in odl when doing landweber reconstruction.
            # - Julius Häger 2026-05-06

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
                            progress.value = int(float(m.group(1)))
                            slicer.app.processEvents()
                            #if canceled[0]:
                            if progress.wasCanceled:
                                slicer.util.errorDisplay("Reconstruction canceled.", windowTitle="Reconstruction")
                                return
                        except Exception:
                            pass

                    # Capture explicit output hint: "Output: /images/xxx.nrrd"
                    if "Output:" in line:
                        output_url = line.split("Output:", 1)[1].strip()

            # Determine filename to load from /images
            if output_url:
                filename = output_url.rsplit("/", 1)[-1]
            else:
                filename = f"tree{tree_ID}_disk{disk_ID}_{method}.nrrd"

            volume = stream_nrrd_from_url(filename, base, progress)
            progress.close()

            if volume != None:
                REC_MIN_X, REC_MIN_Y, REC_MIN_Z = params['REC_MIN_X'], params['REC_MIN_Y'], params["REC_MIN_Z"]
                REC_MAX_X, REC_MAX_Y, REC_MAX_Z = params['REC_MAX_X'], params['REC_MAX_Y'], params["REC_MAX_Z"]
                REC_NPX_X, REC_NPX_Y, REC_NPX_Z = params['REC_NPX_X'], params['REC_NPX_Y'], params["REC_NPX_Z"]
                
                REC_SIZE_X, REC_SIZE_Y, REC_SIZE_Z = REC_MAX_X - REC_MIN_X, REC_MAX_Y - REC_MIN_Y, REC_MAX_Z - REC_MIN_Z
                REC_SPACING_X, REC_SPACING_Y, REC_SPACING_Z = REC_SIZE_X / REC_NPX_X, REC_SIZE_Y / REC_NPX_Y, REC_SIZE_Z / REC_NPX_Z

                print(f"Origin: {(REC_MIN_X, REC_MIN_Y, REC_MIN_Z)}")
                print(f"Spacing: {(REC_SPACING_X, REC_SPACING_Y, REC_SPACING_Z)}")

                volume.SetOrigin(REC_MIN_X, REC_MIN_Y, REC_MIN_Z)
                volume.SetSpacing(REC_SPACING_X, REC_SPACING_Y, REC_SPACING_Z)

                volume.SetIJKToRASDirections([(1,0,0), (0,1,0), (0,0,1)])

        except requests.exceptions.RequestException as e:
            print(traceback.format_exc())
            slicer.util.errorDisplay(
                f"Failed to start reconstruction at:\n{run_url}\n\n{e}",
                windowTitle="Reconstruction Error"
            )
        except Exception as e:
            print(traceback.format_exc())
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
            self.sampleData.sinogram_min_value = response_json["sinogram_min"]
            self.sampleData.sinogram_max_value = response_json["sinogram_max"]
            self.sampleData.sinogram_shape = tuple(response_json["sinogram_shape"])
            print("shape ", self.sampleData.sinogram_shape)
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

            self.ui.sinogramWidget.setEnabled(True)
            self.ui.reconstructionWidget.setEnabled(True)
            self.ui.roiWidget.setEnabled(True)
            self.ui.sinogramRangeWidget.setRange(0, self.sampleData.totalSamples - 1)

            self.sampleData.metadata = response_json["metadata"]

            self.sampleData.rec_bounds_min = np.array([self.sampleData.metadata["REC_MIN_X"], self.sampleData.metadata["REC_MIN_Y"], self.sampleData.bounds_min[2]])
            self.sampleData.rec_bounds_max = np.array([self.sampleData.metadata["REC_MAX_X"], self.sampleData.metadata["REC_MAX_Y"], self.sampleData.bounds_max[2]])

            self.ui.metadataGroupBox.visible = True
            self.ui.metadataTableWidget.clear()
            self.ui.metadataTableWidget.setRowCount(0)
            self.ui.metadataTableWidget.setHorizontalHeaderLabels(["Name", "Value"])
            self.ui.metadataTableWidget.sortingEnabled = False
            for key, value in self.sampleData.metadata.items():
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
            self.ui.roiResolutionCoordinateWidget.setMRMLScene(slicer.mrmlScene)

            sample_roi_dir = Path(os.path.expanduser(f"~/Documents/SinoRecons/{self.sampleData.specie}_{self.sampleData.tree_ID}_{self.sampleData.disk_ID}/"))
            self.clearROIs()
            self.loadROIsForSample(sample_roi_dir)
            # Start with no ROI selected.
            self.ui.roiListWidget.setCurrentRow(-1)

            # Load first frame
            self.onIndexChanged(0, synchronous=True)
            slicer.app.layoutManager().sliceWidget("Gray").fitSliceToBackground()
            
            self.setImageOnSensor(self.sceneObjects.sourceDetectorObjects, self.ui.showSinogramOnSensorCheckbox.checkState())
            self.setSourceDetectorVisible(self.sceneObjects.sourceDetectorObjects, self.ui.showSourceDetectorCheckBox.checked)

            self.setImageOnSensor(self.sceneObjects.sinogramRangeStartSourceDetector, self.ui.showSinogramOnSensorCheckbox.checkState())
            self.setImageOnSensor(self.sceneObjects.sinogramRangeEndSourceDetector, self.ui.showSinogramOnSensorCheckbox.checkState())
            
        except Exception as e:
            print(traceback.format_exc())
            slicer.util.errorDisplay(f"Failed to load full dataset:\n{e}")

    def showROIs(self, state : int):
        if state == qt.Qt.Checked:
            self.settings.setValue(SETTINGS_KEY_SHOW_REGIONS_OF_INTEREST, True)
            for row in range(self.ui.roiListWidget.count):
                item = self.ui.roiListWidget.item(row)
                itemData: ROIData = item.data(qt.Qt.UserRole)
                itemData.roi_node.GetMarkupsDisplayNode().Visibility3DOn()
        else:
            self.settings.setValue(SETTINGS_KEY_SHOW_REGIONS_OF_INTEREST, False)
            for row in range(self.ui.roiListWidget.count):
                item = self.ui.roiListWidget.item(row)
                itemData: ROIData = item.data(qt.Qt.UserRole)
                itemData.roi_node.GetMarkupsDisplayNode().Visibility3DOff()

    def showROISinogramRangeSourceDetectorStateChanged(self, state: int):
        visible = state == qt.Qt.Checked
        self.settings.setValue(SETTINGS_KEY_SHOW_REGIONS_OF_INTEREST_SOURCE_DETECTOR, visible)
        if self.ui.roiListWidget.currentRow != -1:
            item = self.ui.roiListWidget.item(self.ui.roiListWidget.currentRow)
            itemData: ROIData = item.data(qt.Qt.UserRole)
            if self.sceneObjects.sinogramRangeStartSourceDetector.sensorModelImage.GetPointData().GetScalars() == None:
                slice_data = self.fetchSinogramSliceFast(itemData.sinogram_start_index)
                tex_data = (np.iinfo(np.uint8).max * (slice_data - self.sampleData.sinogram_min_value) / (self.sampleData.sinogram_max_value - self.sampleData.sinogram_min_value)).astype(np.uint8)
                self.setSensorImageData(self.sceneObjects.sinogramRangeStartSourceDetector, tex_data)
            if self.sceneObjects.sinogramRangeEndSourceDetector.sensorModelImage.GetPointData().GetScalars() == None:
                slice_data = self.fetchSinogramSliceFast(itemData.sinogram_end_index)
                tex_data = (np.iinfo(np.uint8).max * (slice_data - self.sampleData.sinogram_min_value) / (self.sampleData.sinogram_max_value - self.sampleData.sinogram_min_value)).astype(np.uint8)
                self.setSensorImageData(self.sceneObjects.sinogramRangeStartSourceDetector, tex_data)
        self.setSourceDetectorVisible(self.sceneObjects.sinogramRangeStartSourceDetector, visible)
        self.setSourceDetectorVisible(self.sceneObjects.sinogramRangeEndSourceDetector, visible)

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

        def setEditorData(self, editor: qt.QWidget, index: qt.QModelIndex) -> None:
            roi_data: ROIData = index.data(qt.Qt.UserRole)
            user_property = editor.metaObject().userProperty().name()
            editor.setProperty(user_property, roi_data.name)

        def paint(self, painter: qt.QPainter, option: qt.QStyleOptionViewItem, index: qt.QModelIndex):
            # In the Qt source code QListView only sets State_HasFocus if the list also has focus. See: qlistview.cpp QListView::paintEvent
            # Which means that the highlight on the currently selected item is no longer highlighted when the list looses focus.
            # To fix this we detect the case where the list doesn't have focus and the index we are drawing atm is the selected one
            # and add the State_HasFocus flag to the style options.
            # - Julius Häger 2026-05-21
            if self.parent().focus == False and index == self.parent().selectionModel().currentIndex:
                option.state |= qt.QStyle.State_HasFocus
            qt.QStyledItemDelegate.paint(self, painter, option, index)

    def addNewROIClicked(self):
        self.addROI()

    def addROI(self,
               name: str|None = None,
               id: uuid.UUID|None = None,
               center: vtk.vtkVector3f|None = None,
               size: vtk.vtkVector3f|None = None,
               sinogram_start: int|None = None,
               sinogram_end: int|None = None,
               resolution: tuple[int, int, int]|None = None,
               auto_resolution: bool|None = None,
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
        else:
            default_center = (self.sampleData.rec_bounds_min + self.sampleData.rec_bounds_max) * 0.5
            print(default_center)
            roi_data.roi_node.SetCenter(vtk.vtkVector3f(default_center[0], default_center[1], default_center[2]))
        
        if size is not None:
            roi_data.roi_node.SetSize((size.GetX(), size.GetY(), size.GetZ()))
        else:
            default_size = self.sampleData.rec_bounds_max - self.sampleData.rec_bounds_min
            roi_data.roi_node.SetSize(default_size)

        if resolution is not None:
            roi_data.resolution = resolution
        else:
            # Default resolution
            curr_size = roi_data.roi_node.GetSize()
            rec_pic_size = self.sampleData.metadata['REC_PIC_SIZE']
            roi_data.resolution = (int(curr_size[0] // rec_pic_size), int(curr_size[1] // rec_pic_size), int(curr_size[2] // rec_pic_size))
            print(f"Default resolution: {roi_data.resolution}")

        if auto_resolution is not None:
            roi_data.auto_resolution = auto_resolution
        else:
            roi_data.auto_resolution = True

        # Update UI with the new roi
        roi_data.roi_node.AddObserver(vtk.vtkCommand.ModifiedEvent, self.roiNodeChanged)
        # FIXME: Some way to detect when the roi node is clicked in the 3D view so we can select it there.
        # Most of the obvious ways don't work like trying to add observers for left mouse button events.
        # slicer.app.layoutManager().threeDWidget(0).threeDView().interactor() gives us these events but it's unclear
        # how to detect what object was clicked...
        # - Julius Häger 2026-05-06
    
        roi_data.sinogram_start_index = 0 if sinogram_start is None else sinogram_start
        roi_data.sinogram_end_index = self.sampleData.totalSamples - 1 if sinogram_end is None else sinogram_end

        roi_data.roi_list_widget = qt.QListWidgetItem()
        roi_data.roi_list_widget.setText(name)
        roi_data.roi_list_widget.setFlags(qt.Qt.ItemIsEditable | qt.Qt.ItemIsEnabled)
        roi_data.roi_list_widget.setData(qt.Qt.UserRole, roi_data)

        self.setInteractive(roi_data.roi_list_widget, False)

        if self.ui.showROICheckbox.checkState() != qt.Qt.Checked:
            roi_data.roi_node.GetMarkupsDisplayNode().Visibility3DOff()

        # To be able to use QListWidgetItemModifiedDelegate we need to load
        slicer.packaging.pip_ensure("pathvalidate", requester="SinoReconsVisual2",skip_in_testing = False)

        self.ui.roiListWidget.addItem(roi_data.roi_list_widget)
        self.ui.roiListWidget.setCurrentItem(roi_data.roi_list_widget)
        self.ui.roiListWidget.setItemDelegate(self.QListWidgetItemModifiedDelegate(self.ui.roiListWidget))

        self.ui.reconstructionROIComboBox.addItem(roi_data.name, roi_data)

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
        self.ui.reconstructionROIComboBox.clear()
        for row in reversed(range(self.ui.roiListWidget.count)):
            item = self.ui.roiListWidget.takeItem(row)
            itemData: ROIData = item.data(qt.Qt.UserRole)
            slicer.mrmlScene.RemoveNode(itemData.roi_node)

    def removeROIClicked(self):
        if self.ui.roiListWidget.currentRow != -1:
            item = self.ui.roiListWidget.item(self.ui.roiListWidget.currentRow)
            self.removeROI(item)
            
    def removeROI(self, item: qt.QListWidgetItem):
        if slicer.util.confirmOkCancelDisplay(f"Delete '{item.data(qt.Qt.UserRole).name}'?", windowTitle="Delete ROI?") == False:
            return

        self.ui.roiListWidget.takeItem(self.ui.roiListWidget.row(item))
        itemData: ROIData = item.data(qt.Qt.UserRole)
        slicer.mrmlScene.RemoveNode(itemData.roi_node)

        i = self.ui.reconstructionROIComboBox.findData(itemData)
        if i != -1:
            self.ui.reconstructionROIComboBox.removeItem(i)

        if itemData.original_path != None:
            slicer.packaging.pip_ensure("Send2Trash", requester="SinoReconsVisual2")
            try:
                import send2trash
                send2trash.send2trash(itemData.original_path)
            except:
                # If we can't send to trash, delete the item.
                # FIXME: Confirmation dialog??
                os.remove(itemData.original_path)

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
            resolution = itemData.resolution

            try:
                self.ui.roiCenterCoordinateWidget.blockSignals(True)
                self.ui.roiCenterCoordinateWidget.setCoordinates(center.GetX(), center.GetY(), center.GetZ(), 0)
                self.ui.roiCenterCoordinateWidget.blockSignals(False)

                self.ui.roiSizeCoordinateWidget.blockSignals(True)
                self.ui.roiSizeCoordinateWidget.setCoordinates(size[0], size[1], size[2], 0)
                self.ui.roiSizeCoordinateWidget.blockSignals(False)

                self.ui.roiResolutionCoordinateWidget.blockSignals(True)
                self.ui.roiResolutionCoordinateWidget.setCoordinates(resolution[0], resolution[1], resolution[2], 0)
                self.ui.roiResolutionCoordinateWidget.blockSignals(False)
            except:
                # Current version of slicer doesn't have the setCoordinate PR merged yet.
                # See: https://github.com/commontk/CTK/pull/1417
                self.ui.roiCenterCoordinateWidget.blockSignals(True)
                self.writeValuesToCoordinateWidget(self.ui.roiCenterCoordinateWidget, [center.GetX(),center.GetY(),center.GetZ()])
                self.ui.roiCenterCoordinateWidget.blockSignals(False)

                self.ui.roiSizeCoordinateWidget.blockSignals(True)
                self.writeValuesToCoordinateWidget(self.ui.roiSizeCoordinateWidget, size)
                self.ui.roiSizeCoordinateWidget.blockSignals(False)

                self.ui.roiResolutionCoordinateWidget.blockSignals(True)
                self.writeValuesToCoordinateWidget(self.ui.roiResolutionCoordinateWidget, resolution)
                self.ui.roiResolutionCoordinateWidget.blockSignals(False)

            self.ui.sinogramRangeWidget.blockSignals(True)
            self.ui.sinogramRangeWidget.setValues(itemData.sinogram_start_index, itemData.sinogram_end_index)
            self.ui.sinogramRangeWidget.blockSignals(False)
            self.updateRoiSinogramRange(itemData.sinogram_start_index, itemData.sinogram_end_index, True)
            self.updateROIMetadataLabel(self.ui.roiMetadataLabel, itemData)

            self.ui.roiResolutionCoordinateWidget.enabled = not itemData.auto_resolution

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
            # slicer is weird with their color naming scheme
            # SelectedColor is the default color for ROIs
            # Color is the unselected color which is seemingly unused
            # ActiveColor is the color that shows when the user hovers the ROI
            # - Julius Häger 2026-05-20
            color = self.settings.value("SinoReconsVisual2/ROI/SelectedColor")
            roiDisplayNode.SetSelectedColor(color.redF(), color.greenF(), color.blueF())
            roiDisplayNode.SetActiveColor(color.redF(), color.greenF(), color.blueF())
            roiDisplayNode.SetFillOpacity(color.alphaF() * 0.8) # FIXME: Separate setting for this?
            roiDisplayNode.SetOutlineOpacity(color.alphaF())

            roiDisplayNode.SetHandlesInteractive(True)
            roiDisplayNode.SetTranslationHandleVisibility(False)
            roiDisplayNode.SetScaleHandleVisibility(True)

            print(f"selectable: {roiDisplayNode.GetSelectable()} selected {roiDisplayNode.GetSelected()}")

            for i in range(roiNode.GetNumberOfControlPoints()):
                roiNode.SetNthControlPointVisibility(i, True)
        else:
            color = self.settings.value("SinoReconsVisual2/ROI/InactiveColor")
            roiDisplayNode.SetSelectedColor(color.redF(), color.greenF(), color.blueF())
            roiDisplayNode.SetActiveColor(color.redF(), color.greenF(), color.blueF())
            roiDisplayNode.SetFillOpacity(color.alphaF() * 0.8) # FIXME: Separate setting for this?
            roiDisplayNode.SetOutlineOpacity(color.alphaF())

            roiDisplayNode.SetHandlesInteractive(False)
            
            for i in range(roiNode.GetNumberOfControlPoints()):
                roiNode.SetNthControlPointVisibility(i, False)

    def _getSampleDirectory(self) -> Path:
        return Path(os.path.expanduser(f"~/Documents/SinoRecons/{self.sampleData.specie}_{self.sampleData.tree_ID}_{self.sampleData.disk_ID}"))
    
    def saveROIClicked(self):
        if self.ui.roiListWidget.currentRow != -1:
            item = self.ui.roiListWidget.item(self.ui.roiListWidget.currentRow)
            itemData: ROIData = item.data(qt.Qt.UserRole)
            
            sample_dir = self._getSampleDirectory()
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
        data["resolution"] = roi.resolution
        data["auto_resolution"] = roi.auto_resolution

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
        resolution = (int(data["resolution"][0]), int(data["resolution"][1]), int(data["resolution"][2]))
        auto_resolution = bool(data["auto_resolution"])
        
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
            roi = self.addROI(name, id, center, size, sinogram_start_index, sinogram_end_index, resolution, auto_resolution, path)
            self.roiUpdateModified(roi, False)

    def resetROIFromFile(self, roi: ROIData):
        if (roi.original_path == None):
            print(f"Trying to reset roi '{roi.name}' from file, but it has never been saved to file. This is a plugin error as this should never be allowed to happen.")
            return
        path : Path = roi.original_path
        data: dict[str, typing.Any]
        with open(path, "r") as f:
            data = json.load(f)
        version = int(data["version"])
        if version != 0:
            raise VersionNotSupportedError("The ROI file version is newer than this plugin understands. Maybe there is a new plugin version?", version)
        assert roi.uuid == uuid.UUID(data["uuid"])
        roi.name = path.stem
        roi.roi_node.SetName(roi.name)
        roi.roi_node.SetCenter(vtk.vtkVector3f(data["center"]))
        roi.roi_node.SetSize(vtk.vtkVector3f(data["size"]))
        roi.sinogram_start_index = int(data["sinogram_range"][0])
        roi.sinogram_end_index = int(data["sinogram_range"][1])
        roi.resolution = (int(data["resolution"][0]), int(data["resolution"][1]), int(data["resolution"][2]))
        # FIXME: Update the UI if this roi is the selected roi.
        if self.ui.roiListWidget.item(self.ui.roiListWidget.currentRow).data(qt.Qt.UserRole) == roi:
            center = roi.roi_node.GetCenter()
            size = roi.roi_node.GetSize()
            resolution = roi.resolution
            try:
                self.ui.roiCenterCoordinateWidget.blockSignals(True)
                self.ui.roiCenterCoordinateWidget.setCoordinates(center.GetX(), center.GetY(), center.GetZ(), 0)
                self.ui.roiCenterCoordinateWidget.blockSignals(False)

                self.ui.roiSizeCoordinateWidget.blockSignals(True)
                self.ui.roiSizeCoordinateWidget.setCoordinates(size[0], size[1], size[2], 0)
                self.ui.roiSizeCoordinateWidget.blockSignals(False)

                self.ui.roiResolutionCoordinateWidget.blockSignals(True)
                self.ui.roiResolutionCoordinateWidget.setCoordinates(resolution[0], resolution[1], resolution[2], 0)
                self.ui.roiResolutionCoordinateWidget.blockSignals(False)
            except:
                # Current version of slicer doesn't have the setCoordinate PR merged yet.
                # See: https://github.com/commontk/CTK/pull/1417
                self.ui.roiCenterCoordinateWidget.blockSignals(True)
                self.writeValuesToCoordinateWidget(self.ui.roiCenterCoordinateWidget, [center.GetX(),center.GetY(),center.GetZ()])
                self.ui.roiCenterCoordinateWidget.blockSignals(False)

                self.ui.roiSizeCoordinateWidget.blockSignals(True)
                self.writeValuesToCoordinateWidget(self.ui.roiSizeCoordinateWidget, size)
                self.ui.roiSizeCoordinateWidget.blockSignals(False)

                self.ui.roiResolutionCoordinateWidget.blockSignals(True)
                self.writeValuesToCoordinateWidget(self.ui.roiResolutionCoordinateWidget, resolution)
                self.ui.roiResolutionCoordinateWidget.blockSignals(False)

            self.ui.sinogramRangeWidget.blockSignals(True)
            self.ui.sinogramRangeWidget.setValues(roi.sinogram_start_index, roi.sinogram_end_index)
            self.ui.sinogramRangeWidget.blockSignals(False)
            self.updateRoiSinogramRange(roi.sinogram_start_index, roi.sinogram_end_index, True)

        self.roiUpdateModified(roi, False)


    # FIXME: This is a workaround for ctkCoordinateWidget not having
    # any good python bindings. Here we are hoping that the children of the coordinate widget
    # are sorted in axis order.
    # https://github.com/commontk/CTK/pull/1417 adds better bindings for ctkCoordinateWidget
    def readValuesFromCoordinateWidget(self, coordinateWidget) -> list[float]:
        spinBoxes = coordinateWidget.children()
        return [sb.value for sb in spinBoxes if type(sb) == ctk.ctkDoubleSpinBox]

    def writeValuesToCoordinateWidget(self, coordinateWidget, coords: Iterable[float]):
        # FIXME: Access the spinBoxes directly...
        coordinateWidget.coordinates = str.join(',', [str(c) for c in coords])

    # PythonQt seems to bind the parameter as a 'double'
    # instead of a 'double*' meaning the parameter is
    # completely useless in python.
    # - Julius Häger 2026-04-07
    # https://github.com/commontk/CTK/pull/1417 updates this and adds a x,y,z,w overload
    # - Julius Häger 2026-04-29
    def roiCenterChangedOld(self, _broken):
        x,y,z = self.readValuesFromCoordinateWidget(self.ui.roiCenterCoordinateWidget)
        print("center old", x,y,z)
        self.roiCenterChanged(x, y, z, 0)

    def roiCenterChanged(self, x: float, y: float, z: float, w: float):
        if self.ui.roiListWidget.currentRow == -1:
            return
        
        item = self.ui.roiListWidget.item(self.ui.roiListWidget.currentRow)
        itemData: ROIData = item.data(qt.Qt.UserRole)
        roiNode = itemData.roi_node

        roiNode.SetCenter([x, y, z])
        self.roiUpdateModified(itemData, True)

    def roiSizeChangedOld(self, _broken):
        x,y,z = self.readValuesFromCoordinateWidget(self.ui.roiSizeCoordinateWidget)
        print("size old", x, y, z)
        self.roiSizeChanged(x, y, z, 0)

    def roiSizeChanged(self, x: float, y: float, z: float, w: float):
        if self.ui.roiListWidget.currentRow == -1:
            return
        
        item = self.ui.roiListWidget.item(self.ui.roiListWidget.currentRow)
        itemData: ROIData = item.data(qt.Qt.UserRole)
        roiNode = itemData.roi_node

        roiNode.SetSize([x, y, z])

        if itemData.auto_resolution:
            self.roiUpdateResolution(itemData)

        self.roiUpdateModified(itemData, True)

    def roiResolutionChangedOld(self, _broken):
        x,y,z = self.readValuesFromCoordinateWidget(self.ui.roiResolutionCoordinateWidget)
        print("res old", x, y, z)
        self.roiResolutionChanged(x, y, z, 0)

    def roiResolutionChanged(self, x: float, y: float, z: float, w: float):
        if self.ui.roiListWidget.currentRow == -1:
            return
        
        item = self.ui.roiListWidget.item(self.ui.roiListWidget.currentRow)
        itemData: ROIData = item.data(qt.Qt.UserRole)
        
        itemData.resolution = (int(x), int(y), int(z))

        self.updateROIMetadataLabel(itemData)
        
        self.roiUpdateModified(itemData, True)

    def roiUpdateResolution(self, roi: ROIData):
        assert roi.auto_resolution
        curr_size = roi.roi_node.GetSize()
        rec_pic_size = self.sampleData.metadata['REC_PIC_SIZE']
        roi.resolution = (int(curr_size[0] // rec_pic_size), int(curr_size[1] // rec_pic_size), int(curr_size[2] // rec_pic_size))
        self.writeValuesToCoordinateWidget(self.ui.roiResolutionCoordinateWidget, roi.resolution)

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

        roi = self.getROIDataFromNode(roiNode)
        assert roi != None
        
        try:
            # FIXME: Very inefficient to set the coordinates through strings
            # FIXME: The coordinate string that is returned is rounded to the number of decimals used to display
            # which means we will never be able to accurately get the values from the control... so the modified check will fail.
            self.ui.roiCenterCoordinateWidget.blockSignals(True)
            old_center = [self.ui.roiCenterCoordinateWidget.getCoordinate(0), self.ui.roiCenterCoordinateWidget.getCoordinate(1), self.ui.roiCenterCoordinateWidget.getCoordinate(2)]
            if old_center != [center.GetX(),center.GetY(),center.GetZ()]:
                print(f"change center! {old_center} -> {[center.GetX(),center.GetY(),center.GetZ()]}")
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
        except:
            # Current version of slicer doesn't have the setCoordinate PR merged yet.
            # See: https://github.com/commontk/CTK/pull/1417

            self.ui.roiCenterCoordinateWidget.blockSignals(True)
            old_center = self.readValuesFromCoordinateWidget(self.ui.roiCenterCoordinateWidget)
            if old_center != [center.GetX(),center.GetY(),center.GetZ()]:
                print(f"change center! {old_center} -> {[center.GetX(),center.GetY(),center.GetZ()]}")
                self.writeValuesToCoordinateWidget(self.ui.roiCenterCoordinateWidget, [center.GetX(),center.GetY(),center.GetZ()])
                modified = True
            self.ui.roiCenterCoordinateWidget.blockSignals(False)

            self.ui.roiSizeCoordinateWidget.blockSignals(True)
            old_size = self.readValuesFromCoordinateWidget(self.ui.roiSizeCoordinateWidget)
            if old_size != [size[0],size[1],size[2]]:
                print(f"change size! {old_size} -> {[size[0],size[1],size[2]]}")
                self.writeValuesToCoordinateWidget(self.ui.roiSizeCoordinateWidget, size)
                if roi.auto_resolution:
                    self.roiUpdateResolution(roi)
                modified = True
            self.ui.roiSizeCoordinateWidget.blockSignals(False)
            
        if modified:
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
        self.updateROIMetadataLabel(itemData)

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

    def updateROIMetadataLabel(self, label: qt.QLabel, roi: ROIData):
        size_of_float32 = 4
        sinogram_slice_size = self.sampleData.sinogram_shape[1] * self.sampleData.sinogram_shape[2] * size_of_float32
        input_size: int = (roi.sinogram_end_index - roi.sinogram_start_index) * sinogram_slice_size
        output_size = roi.resolution[0] * roi.resolution[1] * roi.resolution[2] * size_of_float32

        def format_bytes(size):
            # 2**10 = 1024
            power = 2**10
            n = 0
            power_labels = {0 : 'B', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
            while size > power:
                size /= power
                n += 1
            return f"{size:.{1}f}{power_labels[n]}"

        label.text = f"Input sinogram size: {format_bytes(input_size)}, Output volume size: {format_bytes(output_size)}"

    def toggleROIAutoResolution(self, roi: ROIData, enable: bool):
        if roi.auto_resolution != enable:
            self.roiUpdateModified(roi, True)
        roi.auto_resolution = enable
        if self.ui.roiListWidget.item(self.ui.roiListWidget.currentRow).data(qt.Qt.UserRole) == roi:
            self.ui.roiResolutionCoordinateWidget.enabled = not enable

    def showROIItemContextMenu(self, pos: qt.QPoint) -> None:
        item = self.ui.roiListWidget.itemAt(pos)
        if item == None:
            return
        itemData: ROIData = item.data(qt.Qt.UserRole)

        menu: qt.QMenu = qt.QMenu()

        rename = qt.QAction("Rename...", menu)
        rename.triggered.connect(lambda _: self.ui.roiListWidget.editItem(item))
        menu.addAction(rename)

        save = qt.QAction("Save", menu)
        save.triggered.connect(lambda _: self.saveROIToFile(itemData, self._getSampleDirectory()))
        menu.addAction(save)

        def openInFolder(_):
            if itemData.original_path != None:
                # FIXME: show-in-file-manager is broken on windows when installed with slicer.packaging.pip_ensure
                # So we disable this action on windows.
                # See: https://github.com/damonlynch/showinfilemanager/issues/39
                # - Julius Häger 2026-05-19
                if sys.platform == 'win32':
                    import pathvalidate, subprocess
                    try:
                        pathvalidate.validate_filepath(itemData.original_path, platform=pathvalidate.Platform.WINDOWS)
                        subprocess.Popen(["C:\\Windows\\explorer.exe", str(itemData.original_path.parent)], shell=True)
                    except pathvalidate.ValidationError as e:
                        logging.warning(f"Invalid path {e} can't open folder")
                        return
                else:
                    slicer.packaging.pip_ensure("show-in-file-manager==1.1.6", requester="SinoReconsVisual2")
                    from showinfm.showinfm import show_in_file_manager
                    print(str(itemData.original_path))
                    show_in_file_manager("file://" + str(itemData.original_path))

        openFolder = qt.QAction("Open containing folder", menu)
        openFolder.enabled = itemData.original_path != None
        openFolder.triggered.connect(openInFolder)
        menu.addAction(openFolder)

        menu.addSeparator()

        resetChanges = qt.QAction("Discard changes", menu)
        resetChanges.enabled = itemData.original_path != None
        resetChanges.triggered.connect(lambda _: self.resetROIFromFile(itemData))
        menu.addAction(resetChanges)

        autoResolutionToggle = qt.QAction("Auto Resolution", menu)
        autoResolutionToggle.checkable = True
        autoResolutionToggle.checked = itemData.auto_resolution
        autoResolutionToggle.toggled.connect(lambda checked: self.toggleROIAutoResolution(itemData, checked))
        menu.addAction(autoResolutionToggle)

        # FIXME: Add set appropriate resolution action
        # FIXME: Add reset to default action.

        menu.addSeparator()

        delete = qt.QAction("Delete", menu)
        delete.triggered.connect(lambda _: self.removeROI(item))
        menu.addAction(delete)

        menu.exec(self.ui.roiListWidget.mapToGlobal(pos))

    def reconstructionROIChanged(self, index: int):
        if index != -1:
            itemData: ROIData = self.ui.reconstructionROIComboBox.itemData(index)
            self.updateROIMetadataLabel(self.ui.reconstructionROIMetadataLabel, itemData)

    # -----------------------------
    # Rendering helpers
    # -----------------------------
    def createSceneObjects(self):
        sceneObjects = Vtk3DSceneObjects()

        def createSourceDetectorObjects(name_prefix: str) -> SourceDetectorObjects:
            sourceDetector = SourceDetectorObjects()

            sourceDetector.sourceModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name_prefix + "Source")
            polys = vtk.vtkPolyData()
            polys.SetPoints(vtk.vtkPoints())
            polys.SetVerts(vtk.vtkCellArray())
            sourceDetector.sourceModel.SetAndObservePolyData(polys)
            sourceDetector.sourceModelDisplay = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
            sourceDetector.sourceModel.SetAndObserveDisplayNodeID(sourceDetector.sourceModelDisplay.GetID())
            sourceDetector.sourceModelDisplay.SetPointSize(8)
            sourceDetector.sourceModelDisplay.SetVisibility(1)
            color = self.settings.value("SinoReconsVisual2/SourceColor")
            sourceDetector.sourceModelDisplay.SetColor((color.redF(), color.greenF(), color.blueF()))
            sourceDetector.sourceModelDisplay.SetOpacity(color.alphaF())

            sourceDetector.fovRaysModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name_prefix + "FOVRays")
            polys = vtk.vtkPolyData()
            polys.SetPoints(vtk.vtkPoints())
            polys.SetLines(vtk.vtkCellArray())
            sourceDetector.fovRaysModel.SetAndObservePolyData(polys)
            sourceDetector.fovRaysModelDisplay = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
            sourceDetector.fovRaysModel.SetAndObserveDisplayNodeID(sourceDetector.fovRaysModelDisplay.GetID())
            sourceDetector.fovRaysModelDisplay.SetLineWidth(2)
            sourceDetector.fovRaysModelDisplay.SetVisibility(1)
            color = self.settings.value("SinoReconsVisual2/FOVRayColor")
            sourceDetector.fovRaysModelDisplay.SetColor((color.redF(), color.greenF(), color.blueF()))
            sourceDetector.fovRaysModelDisplay.SetOpacity(color.alphaF())

            sourceDetector.sensorModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name_prefix + "Detector")
            polys = vtk.vtkPolyData()
            polys.SetPoints(vtk.vtkPoints())
            polys.SetPolys(vtk.vtkCellArray())
            sourceDetector.sensorModel.SetAndObservePolyData(polys)
            sourceDetector.sensorModelDisplay = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
            sourceDetector.sensorModel.SetAndObserveDisplayNodeID(sourceDetector.sensorModelDisplay.GetID())
            sourceDetector.sensorModelDisplay.SetVisibility(1)
            sourceDetector.sensorModelDisplay.SetRepresentation(slicer.vtkMRMLDisplayNode.SurfaceRepresentation)
            sourceDetector.sensorModelDisplay.SetEdgeVisibility(False)      # Hide edges
            sourceDetector.sensorModelDisplay.SetLighting(0)                # Disable lighting
            sourceDetector.sensorModelDisplay.SetBackfaceCulling(False)     # Render both sides
            sourceDetector.sensorModelImage = vtk.vtkImageData()
            sourceDetector.sensorModelImageProducer = vtk.vtkTrivialProducer()
            sourceDetector.sensorModelImageProducer.SetOutput(sourceDetector.sensorModelImage)
            color = self.settings.value("SinoReconsVisual2/DetectorColor")
            sourceDetector.sensorModelDisplay.SetColor((color.redF(), color.greenF(), color.blueF()))
            sourceDetector.sensorModelDisplay.SetOpacity(color.alphaF())

            return sourceDetector

        sceneObjects.sourceDetectorObjects = createSourceDetectorObjects("")

        sceneObjects.trajectoryModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "Trajectory")
        polys = vtk.vtkPolyData()
        polys.SetPoints(vtk.vtkPoints())
        polys.SetLines(vtk.vtkCellArray())
        sceneObjects.trajectoryModel.SetAndObservePolyData(polys)
        sceneObjects.trajectoryModelDisplay = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        sceneObjects.trajectoryModel.SetAndObserveDisplayNodeID(sceneObjects.trajectoryModelDisplay.GetID())
        sceneObjects.trajectoryModelDisplay.SetLineWidth(2)
        sceneObjects.trajectoryModelDisplay.SetVisibility(1)
        color = self.settings.value("SinoReconsVisual2/TrajectoryColor")
        sceneObjects.trajectoryModelDisplay.SetColor((color.redF(), color.greenF(), color.blueF()))
        sceneObjects.trajectoryModelDisplay.SetOpacity(color.alphaF())

        sceneObjects.reconCubeModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "Reconstruction bounds")
        sceneObjects.reconOutline = vtk.vtkOutlineSource()
        sceneObjects.reconCubeModel.SetPolyDataConnection(sceneObjects.reconOutline.GetOutputPort())
        sceneObjects.reconCubeModelDisplay = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        sceneObjects.reconCubeModel.SetAndObserveDisplayNodeID(sceneObjects.reconCubeModelDisplay.GetID())
        sceneObjects.reconCubeModelDisplay.SetVisibility(0)
        color = self.settings.value("SinoReconsVisual2/BoundsColor")
        sceneObjects.reconCubeModelDisplay.SetColor((color.redF(), color.greenF(), color.blueF()))
        sceneObjects.reconCubeModelDisplay.SetOpacity(color.alphaF())

        grayViewNodeID = slicer.app.layoutManager().sliceWidget("Gray").mrmlSliceNode().GetID()

        # For the sinogram outline we project a cube outline to the slice only in the Gray view.
        # - Julius Häger 2026-03-27
        sceneObjects.sinogramOutline = vtk.vtkOutlineSource()
        sceneObjects.sinogramOutline.SetBounds(-3, 3, 0, 50, 0, 50)
        sceneObjects.sinogramOutlineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "Sinogram Outline")
        sceneObjects.sinogramOutlineNode.SetPolyDataConnection(sceneObjects.sinogramOutline.GetOutputPort())
        sceneObjects.sinogramOutlineDisplay = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        sceneObjects.sinogramOutlineNode.SetAndObserveDisplayNodeID(sceneObjects.sinogramOutlineDisplay.GetID())
        sceneObjects.sinogramOutlineDisplay.SetViewNodeIDs([grayViewNodeID])
        sceneObjects.sinogramOutlineDisplay.SetSliceDisplayModeToProjection()
        sceneObjects.sinogramOutlineDisplay.Visibility3DOff()
        sceneObjects.sinogramOutlineDisplay.Visibility2DOn()
        sceneObjects.sinogramOutlineDisplay.VisibilityOff()
        color = self.settings.value("SinoReconsVisual2/SinogramOutlineColor")
        sceneObjects.sinogramOutlineDisplay.SetColor((color.redF(), color.greenF(), color.blueF()))
        sceneObjects.sinogramOutlineDisplay.SetOpacity(color.alphaF())
        # FIXME: outline color setting.

        sceneObjects.roiSinogramRangeModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "ROI Sinogram Range")
        polys = vtk.vtkPolyData()
        polys.SetPoints(vtk.vtkPoints())
        polys.SetVerts(vtk.vtkCellArray())
        sceneObjects.roiSinogramRangeModel.SetAndObservePolyData(polys)
        sceneObjects.roiSinogramRangeModelDisplay = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        sceneObjects.roiSinogramRangeModel.SetAndObserveDisplayNodeID(sceneObjects.roiSinogramRangeModelDisplay.GetID())
        sceneObjects.roiSinogramRangeModelDisplay.SetPointSize(8)
        sceneObjects.roiSinogramRangeModelDisplay.SetVisibility(1)
        color = self.settings.value("SinoReconsVisual2/ROI/SinogramRangeColor")
        sceneObjects.roiSinogramRangeModelDisplay.SetColor((color.redF(), color.greenF(), color.blueF()))
        sceneObjects.roiSinogramRangeModelDisplay.SetOpacity(color.alphaF())

        sceneObjects.sinogramRangeStartSourceDetector = createSourceDetectorObjects("Sinogram range start ")
        sceneObjects.sinogramRangeEndSourceDetector = createSourceDetectorObjects("Sinogram range end ")

        sceneObjects.roiSinogramTrajectoryModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "ROI Sinogram Trajectory")
        polys = vtk.vtkPolyData()
        polys.SetPoints(vtk.vtkPoints())
        polys.SetLines(vtk.vtkCellArray())
        sceneObjects.roiSinogramTrajectoryModel.SetAndObservePolyData(polys)
        sceneObjects.roiSinogramTrajectoryModelDisplay = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        sceneObjects.roiSinogramTrajectoryModel.SetAndObserveDisplayNodeID(sceneObjects.roiSinogramTrajectoryModelDisplay.GetID())
        sceneObjects.roiSinogramTrajectoryModelDisplay.SetLineWidth(4)
        sceneObjects.roiSinogramTrajectoryModelDisplay.SetVisibility(1)
        color = self.settings.value("SinoReconsVisual2/ROI/SinogramRangeColor")
        sceneObjects.roiSinogramTrajectoryModelDisplay.SetColor((color.redF(), color.greenF(), color.blueF()))
        sceneObjects.roiSinogramTrajectoryModelDisplay.SetOpacity(color.alphaF())

        return sceneObjects

    def destroySceneObjects(self):
        def destroySourceDetectorObjects(sourceDetector: SourceDetectorObjects):
            slicer.mrmlScene.RemoveNode(sourceDetector.sourceModelDisplay)
            slicer.mrmlScene.RemoveNode(sourceDetector.sourceModel)
            slicer.mrmlScene.RemoveNode(sourceDetector.fovRaysModelDisplay)
            slicer.mrmlScene.RemoveNode(sourceDetector.fovRaysModel)
            slicer.mrmlScene.RemoveNode(sourceDetector.sensorModelDisplay)
            slicer.mrmlScene.RemoveNode(sourceDetector.sensorModel)
            # FIXME: Delete the image?
            sourceDetector.sourceModelDisplay = None
            sourceDetector.sourceModel = None
            sourceDetector.fovRaysModelDisplay = None
            sourceDetector.fovRaysModel = None
            sourceDetector.sensorModelDisplay = None
            sourceDetector.sensorModel = None

        if (hasattr(self, 'sceneObjects')) and self.sceneObjects:
            destroySourceDetectorObjects(self.sceneObjects.sourceDetectorObjects)
            slicer.mrmlScene.RemoveNode(self.sceneObjects.trajectoryModelDisplay)
            slicer.mrmlScene.RemoveNode(self.sceneObjects.trajectoryModel)
            slicer.mrmlScene.RemoveNode(self.sceneObjects.reconCubeModelDisplay)
            slicer.mrmlScene.RemoveNode(self.sceneObjects.reconCubeModel)
            slicer.mrmlScene.RemoveNode(self.sceneObjects.sinogramOutlineDisplay)
            slicer.mrmlScene.RemoveNode(self.sceneObjects.sinogramOutlineNode)
            slicer.mrmlScene.RemoveNode(self.sceneObjects.roiSinogramRangeModelDisplay)
            slicer.mrmlScene.RemoveNode(self.sceneObjects.roiSinogramRangeModel)
            destroySourceDetectorObjects(self.sceneObjects.sinogramRangeStartSourceDetector)
            destroySourceDetectorObjects(self.sceneObjects.sinogramRangeEndSourceDetector)
            slicer.mrmlScene.RemoveNode(self.sceneObjects.roiSinogramTrajectoryModelDisplay)
            slicer.mrmlScene.RemoveNode(self.sceneObjects.roiSinogramTrajectoryModel)
            self.sceneObjects.trajectoryModelDisplay = None
            self.sceneObjects.trajectoryModel = None
            self.sceneObjects.reconCubeModelDisplay = None
            self.sceneObjects.reconCubeModel = None
            self.sceneObjects.sinogramOutlineNode = None
            self.sceneObjects.sinogramOutlineDisplay = None
            self.sceneObjects.roiSinogramRangeModelDisplay = None
            self.sceneObjects.roiSinogramRangeModel = None
            self.sceneObjects.roiSinogramTrajectoryModelDisplay = None
            self.sceneObjects.roiSinogramTrajectoryModel = None

    def updateSceneData(self, index: int):
        self.updateTrajectory()
        self.updateReconstructionBounds()
        self.updateSourceDetector(self.sceneObjects.sourceDetectorObjects, index)

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

    def updateReconstructionBounds(self):
        metadata = self.sampleData.metadata

        trajectory = self.sampleData.geometry.get("full_trajectory", np.empty(0))
        print(trajectory.shape)

        REC_MIN_X, REC_MIN_Y, REC_MIN_Z = metadata['REC_MIN_X'], metadata['REC_MIN_Y'], trajectory[0][2]
        REC_MAX_X, REC_MAX_Y, REC_MAX_Z = metadata['REC_MAX_X'], metadata['REC_MAX_Y'], trajectory[-1][2]
        #REC_NPX_X, REC_NPX_Y, REC_NPX_Z = metadata['REC_NPX_X'], metadata['REC_NPX_Y'], int((REC_MAX_Z - REC_MIN_Z) // metadata['REC_PIC_SIZE'])

        self.sceneObjects.reconOutline.SetBounds(REC_MIN_X, REC_MAX_X, REC_MIN_Y, REC_MAX_Y, REC_MIN_Z, REC_MAX_Z)
        self.sceneObjects.reconCubeModelDisplay.SetVisibility(1)

    def updateSourceDetector(self, sourceDetector: SourceDetectorObjects, index: int):
        def updateSource(index: int):
            sources = self.sampleData.geometry.get("sources", np.empty(0))
            source = sources[index].reshape(1, 3)
            #print(f"sources {source.shape} {source.dtype} {len(source)}")

            polyData : vtk.vtkPolyData = typing.cast(vtk.vtkPolyData, sourceDetector.sourceModel.GetPolyData())
            
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

        def updateFOVLines(index: int):
            rays = self.sampleData.geometry.get("fov_rays", np.empty(0))
            rays = rays[index]
            if rays.shape[-1] != 3 or rays.shape[-2] != 2:
                print(f"[ERROR] fov_rays had the wrong shape {rays.shape}, expected (N, 2, 3).")
                return
            
            polyData : vtk.vtkPolyData = typing.cast(vtk.vtkPolyData, sourceDetector.fovRaysModel.GetPolyData())
            
            points = typing.cast(vtk.vtkPoints, polyData.GetPoints())
            points.SetData(numpy_to_vtk(np.array(rays).reshape(len(rays)*2, 3), 1))
            #print(points)

            cells = polyData.GetLines()
            cells.SetData(2, numpy_to_vtkIdTypeArray(np.arange(len(rays)*2), 1))
            #print(cells)

            points.Modified()
            cells.Modified()
            polyData.Modified()

        def updateSensorGeometry(index: int):
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

            polyData : vtk.vtkPolyData = typing.cast(vtk.vtkPolyData, sourceDetector.sensorModel.GetPolyData())
            
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

        updateSource(index)
        updateFOVLines(index)
        updateSensorGeometry(index)

    def setSourceDetectorVisible(self, sourceDetector: SourceDetectorObjects, visible: bool):
        sourceDetector.sourceModelDisplay.SetVisibility(1 if visible else 0)
        sourceDetector.fovRaysModelDisplay.SetVisibility(1 if visible else 0)
        sourceDetector.sensorModelDisplay.SetVisibility(1 if visible else 0)

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

            self.updateSourceDetector(self.sceneObjects.sinogramRangeStartSourceDetector, start_index)
            self.updateSourceDetector(self.sceneObjects.sinogramRangeEndSourceDetector, end_index)

            if self.ui.showROISinogramRangeSourceDetectorCheckBox.checked:
                # FIXME: only fetch if the current index has actually changed!
                start_slice_data = self.fetchSinogramSliceFast(start_index)
                end_slice_data = self.fetchSinogramSliceFast(end_index)
                
                start = time.time()
                start_tex_data = (np.iinfo(np.uint8).max * (start_slice_data - self.sampleData.sinogram_min_value) / (self.sampleData.sinogram_max_value - self.sampleData.sinogram_min_value)).astype(np.uint8)
                end_tex_data = (np.iinfo(np.uint8).max * (end_slice_data - self.sampleData.sinogram_min_value) / (self.sampleData.sinogram_max_value - self.sampleData.sinogram_min_value)).astype(np.uint8)

                self.setSensorImageData(self.sceneObjects.sinogramRangeStartSourceDetector, start_tex_data)
                self.setSensorImageData(self.sceneObjects.sinogramRangeEndSourceDetector, end_tex_data)
                end = time.time()
                print(f"[DEBUG] Updating sinogram range sensor texture took: {end - start} s")

        self.setSourceDetectorVisible(self.sceneObjects.sinogramRangeStartSourceDetector, is_visible)
        self.setSourceDetectorVisible(self.sceneObjects.sinogramRangeEndSourceDetector, is_visible)

        self.sceneObjects.roiSinogramRangeModelDisplay.SetVisibility(1 if is_visible else 0)
        self.sceneObjects.roiSinogramTrajectoryModelDisplay.SetVisibility(1 if is_visible else 0)

    def showSourceDetectorStateChanged(self, state: int):
        if hasattr(self, "sceneObjects") == False or self.sceneObjects == None:
            return
        show = state == qt.Qt.Checked
        self.settings.setValue(SETTINGS_KEY_SHOW_SOURCE_DETECTOR, show)
        self.setSourceDetectorVisible(self.sceneObjects.sourceDetectorObjects, show)

    def showSinogramOnSensorStateChanged(self, state: int):
        if hasattr(self, "sceneObjects") == False or self.sceneObjects == None:
            return
        print(f"show on sensor changed: {state}")
        show = state == qt.Qt.Checked
        self.settings.setValue(SETTINGS_KEY_SHOW_SINOGRAM_ON_SENSOR, show)
        self.setImageOnSensor(self.sceneObjects.sourceDetectorObjects, show)
        self.setImageOnSensor(self.sceneObjects.sinogramRangeStartSourceDetector, show)
        self.setImageOnSensor(self.sceneObjects.sinogramRangeEndSourceDetector, show)


    def setImageOnSensor(self, sourceDetector: SourceDetectorObjects, show: bool):
        if show:
            sourceDetector.sensorModelDisplay.SetTextureImageDataConnection(sourceDetector.sensorModelImageProducer.GetOutputPort())
            sourceDetector.sensorModelDisplay.SetOpacity(1.0)
            sourceDetector.sensorModelDisplay.SetInterpolation(slicer.vtkMRMLDisplayNode.FlatInterpolation)
            sourceDetector.sensorModelDisplay.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseDataScalarRange) # Data range = auto
        else:
            sourceDetector.sensorModelDisplay.SetTextureImageDataConnection(None)
            sourceDetector.sensorModelDisplay.SetOpacity(0.7)
            sourceDetector.sensorModelDisplay.SetInterpolation(slicer.vtkMRMLDisplayNode.PhongInterpolation)

    def setSensorImageData(self, sourceDetector: SourceDetectorObjects, tex_data: np.typing.NDArray[np.uint8]):
        sourceDetector.sensorModelImage.SetDimensions(tex_data.shape[1], tex_data.shape[0], 1)
        # FIXME: Do not allocate a new VTK arrray, update the existing one if possible!
        vtk_array = numpy_to_vtk(tex_data.reshape(tex_data.shape[0] * tex_data.shape[1], 1), 1, vtk.VTK_UNSIGNED_CHAR)
        #print(vtk_array)
        sourceDetector.sensorModelImage.GetPointData().SetScalars(vtk_array)

    # -----------------------------
    # Slider / sinogram fetch
    # -----------------------------
    def onIndexChanged(self, value, synchronous = False):
        start = time.time()
        self.currentIndex = value
        self.ui.sliderIndexLabel.setText(f"Index: {value}")
        self.updateSceneData(value)
        if self.playButtonTimer.isActive() or synchronous:
            self.loadPreviewSlice()
        else:
            self.sliderDebounceTimer.start()
        self.fullDetailDebounceTimer.start()
        end = time.time()
        print(f"[DEBUG] onIndexChanged took {end - start} seconds")

    def loadPreviewSlice(self):
        try:
            sliceData = self.fetchSinogramSliceFast(self.currentIndex)

            start = time.time()
            if hasattr(self, 'volume_node') and self.volume_node:
                voxel_data = slicer.util.arrayFromVolume(self.volume_node)
                #print(f"voxel_data shape: {voxel_data.shape} {voxel_data.dtype}")
                voxel_data[:,:,0] = sliceData
                slicer.util.arrayFromVolumeModified(self.volume_node)
            else:
                self.volume_node = slicer.util.addVolumeFromArray(sliceData[:, :, np.newaxis])
                self.volume_node.name = f"Preview sinogram"

            self.volume_node.GetDisplayNode().SetAutoWindowLevel(False)
            self.volume_node.GetDisplayNode().SetWindowLevelMinMax(self.sampleData.sinogram_min_value, self.sampleData.sinogram_max_value)

            # Update outline bounds.
            bounds = np.empty(6)
            self.volume_node.GetBounds(bounds)
            self.sceneObjects.sinogramOutline.SetBounds(*bounds)
            self.sceneObjects.sinogramOutlineDisplay.VisibilityOn()

            end = time.time()
            print(f"[DEBUG] Adding/modifying volume took: {end - start} s")

            start = time.time()
            tex_data = (np.iinfo(np.uint8).max * (sliceData - self.sampleData.sinogram_min_value) / (self.sampleData.sinogram_max_value - self.sampleData.sinogram_min_value)).astype(np.uint8)
            # FIXME: Do we need to flip here? What is the correct way to show this?
            # Alternatively the UV coordinates on the sensor geometry is "wrong"...
            #tex_data = np.flip(tex_data, axis=1)
            self.setSensorImageData(self.sceneObjects.sourceDetectorObjects, tex_data)
            end = time.time()
            print(f"[DEBUG] Updating sensor texture took: {end - start} s")
            
            if self.volume_node:
                slicer.app.layoutManager().sliceWidget("Gray").sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(self.volume_node.GetID())
            else:
                slicer.util.errorDisplay("Failed to load sinogram slice.")

        except Exception as e:
            self.sliderDebounceTimer.stop()
            import traceback
            #print(e)
            print(traceback.format_exc())
            #slicer.util.errorDisplay(f"Error fetching sinogram slice: {e} {traceback.format_exc()}")

    def fetchSinogramSliceFast(self, index: int) -> np.typing.NDArray[np.float32]:
        base = self._currentBaseUrl()

        start = time.time()
        response_fast = self.session.get(f"{base}/get_sinogram_slice_fast/{index}")
        end = time.time()
        print(f"[DEBUG] Download (fast) took {end-start} s {len(response_fast.content)/1000} kb")

        if response_fast.status_code >= 400:
            print(f"Failed to load sinogram slice: {response_fast.headers}")

        sliceMin = np.float32(response_fast.headers["slice_min"])
        sliceMax = np.float32(response_fast.headers["slice_max"])

        start = time.time()
        img = PIL.Image.open(io.BytesIO(response_fast.content))
        img_data = np.array(img)
        img_data_mapped = img_data * ((sliceMax - sliceMin) / np.float32(255.0)) + sliceMin
        end = time.time()
        print(f"[DEBUG] Decoding image took {end-start} s {img.size} {img.mode} {img_data_mapped.shape} {img_data_mapped.dtype}")
        return img_data_mapped

    def loadFullDetailSlice(self):
        try:
            slicer.packaging.pip_ensure("pynrrd", requester="SinoReconsVisual2")
            import nrrd

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
            self.full_detail_volume_node.GetDisplayNode().SetWindowLevelMinMax(self.sampleData.sinogram_min_value, self.sampleData.sinogram_max_value)
            
            # Update outline bounds.
            bounds = np.empty(6)
            self.full_detail_volume_node.GetBounds(bounds)
            self.sceneObjects.sinogramOutline.SetBounds(*bounds)
            self.sceneObjects.sinogramOutlineDisplay.VisibilityOn()

            end = time.time()
            print(f"[DEBUG] Adding/modifying volume took: {end - start} s")

            if self.full_detail_volume_node:
                slicer.app.layoutManager().sliceWidget("Gray").sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(self.full_detail_volume_node.GetID())
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
    # Settings callbacks
    # -----------------------------
    def sourceTrajectoryColorChanged(self, color: qt.QColor):
        if hasattr(self, 'sceneObjects'):
            self.sceneObjects.trajectoryModelDisplay.SetColor(color.redF(), color.greenF(), color.blueF())
            self.sceneObjects.trajectoryModelDisplay.SetOpacity(color.alphaF())

    def sourceColorChanged(self, color: qt.QColor):
        def setSourceColor(sd: SourceDetectorObjects, color: qt.QColor):
            sd.sourceModelDisplay.SetColor(color.redF(), color.greenF(), color.blueF())
            sd.sourceModelDisplay.SetOpacity(color.alphaF())

        if hasattr(self, 'sceneObjects'):
            setSourceColor(self.sceneObjects.sourceDetectorObjects, color)
            setSourceColor(self.sceneObjects.sinogramRangeStartSourceDetector, color)
            setSourceColor(self.sceneObjects.sinogramRangeEndSourceDetector, color)

    def detectorColorChanged(self, color: qt.QColor):
        def setDetectorColor(sd: SourceDetectorObjects, color: qt.QColor):
            sd.sensorModelDisplay.SetColor(color.redF(), color.greenF(), color.blueF())
            sd.sensorModelDisplay.SetOpacity(color.alphaF())

        if hasattr(self, 'sceneObjects'):    
            setDetectorColor(self.sceneObjects.sourceDetectorObjects, color)
            setDetectorColor(self.sceneObjects.sinogramRangeStartSourceDetector, color)
            setDetectorColor(self.sceneObjects.sinogramRangeEndSourceDetector, color)

    def fovRaysColorChanged(self, color: qt.QColor):
        def setFOVRaysColor(sd: SourceDetectorObjects, color: qt.QColor):
            sd.fovRaysModelDisplay.SetColor(color.redF(), color.greenF(), color.blueF())
            sd.fovRaysModelDisplay.SetOpacity(color.alphaF())

        if hasattr(self, 'sceneObjects'):
            setFOVRaysColor(self.sceneObjects.sourceDetectorObjects, color)
            setFOVRaysColor(self.sceneObjects.sinogramRangeStartSourceDetector, color)
            setFOVRaysColor(self.sceneObjects.sinogramRangeEndSourceDetector, color)

    def boundsColorChanged(self, color: qt.QColor):
        if hasattr(self, 'sceneObjects'):
            self.sceneObjects.reconCubeModelDisplay.SetColor(color.redF(), color.greenF(), color.blueF())
            self.sceneObjects.reconCubeModelDisplay.SetOpacity(color.alphaF())

    def sinogramOutlineColorChanged(self, color: qt.QColor):
        if hasattr(self, 'sceneObjects'):
            self.sceneObjects.sinogramOutlineDisplay.SetColor(color.redF(), color.greenF(), color.blueF())
            self.sceneObjects.sinogramOutlineDisplay.SetOpacity(color.alphaF())

    def selectedROIColorChanged(self, color: qt.QColor):
        if hasattr(self, 'ui'):
            if self.ui.roiListWidget.currentRow != -1:
                item = self.ui.roiListWidget.item(self.ui.roiListWidget.currentRow)
                itemData: ROIData = item.data(qt.Qt.UserRole)
                roiNode = itemData.roi_node
                roiDisplayNode = roiNode.GetMarkupsDisplayNode()
                roiDisplayNode.SetSelectedColor(color.redF(), color.greenF(), color.blueF())
                roiDisplayNode.SetActiveColor(color.redF(), color.greenF(), color.blueF())
                roiDisplayNode.SetFillOpacity(color.alphaF() * 0.8) # FIXME: Separate setting for this?
                roiDisplayNode.SetOutlineOpacity(color.alphaF())

    def inactiveROIColorChanged(self, color: qt.QColor):
        if hasattr(self, 'sceneObjects'):
            for row in range(self.ui.roiListWidget.count):
                if row == self.ui.roiListWidget.currentRow:
                    continue
                item = self.ui.roiListWidget.item(row)
                itemData: ROIData = item.data(qt.Qt.UserRole)
                roiNode = itemData.roi_node
                roiDisplayNode = roiNode.GetMarkupsDisplayNode()
                roiDisplayNode.SetSelectedColor(color.redF(), color.greenF(), color.blueF())
                roiDisplayNode.SetActiveColor(color.redF(), color.greenF(), color.blueF())
                roiDisplayNode.SetFillOpacity(color.alphaF() * 0.8) # FIXME: Separate setting for this?
                roiDisplayNode.SetOutlineOpacity(color.alphaF())

    def roiSinogramRangeColorChanged(self, color: qt.QColor):
        if hasattr(self, 'sceneObjects'):
            self.sceneObjects.roiSinogramTrajectoryModelDisplay.SetColor(color.redF(), color.greenF(), color.blueF())
            self.sceneObjects.roiSinogramTrajectoryModelDisplay.SetOpacity(color.alphaF())
            self.sceneObjects.roiSinogramRangeModelDisplay.SetColor(color.redF(), color.greenF(), color.blueF())
            self.sceneObjects.roiSinogramRangeModelDisplay.SetOpacity(color.alphaF())
        if hasattr(self, 'ui'):
            palette = self.ui.sinogramRangeWidget.palette
            palette.setColor(qt.QPalette.Normal, qt.QPalette.Highlight, color)
            palette.setColor(qt.QPalette.Inactive, qt.QPalette.Highlight, color)
            palette.setColor(qt.QPalette.Disabled, qt.QPalette.Highlight, qt.QColor.fromRgbF(color.redF()*0.55, color.greenF()*0.55, color.blueF()*0.55))
            self.ui.sinogramRangeWidget.palette = palette

# -----------------------------
# HTTP helpers (use saved base URL)
# -----------------------------
def stream_nrrd_from_url(filename, base_url: Optional[str] = None, progress_dialog: qt.QProgressDialog = None) -> slicer.vtkMRMLVolumeNode:

    base = normalize_base_url(base_url or get_saved_base_url())
    url = f"{base}/images/{filename}"
    temp_file_path = os.path.join(slicer.app.temporaryPath, filename)
    
    try:
        response = requests.head(url)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        create_progress_dialog = False
        if progress_dialog == None:
            create_progress_dialog = True
            progress_dialog = slicer.util.createProgressDialog(value=0, maximum=total_size)
        progress_dialog.setLabelText("Loading reconstruction...")
        progress_dialog.setWindowTitle("Load Progress")
        progress_dialog.setMaximum(total_size)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.value = 0

        with requests.get(url, stream=True) as r, open(temp_file_path, "wb") as temp_file:
            r.raise_for_status()
            downloaded_size = 0
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
                    downloaded_size += len(chunk)
                    progress_dialog.value = downloaded_size
                    slicer.app.processEvents()
                    if progress_dialog.wasCanceled:
                        slicer.util.errorDisplay("Download was canceled by the user.", windowTitle="Download Canceled")
                        return None

        volume_node = slicer.util.loadVolume(temp_file_path)
        if volume_node:
            slicer.app.layoutManager().sliceWidget("Gray").sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(volume_node.GetID())
        else:
            print("Failed to load the NRRD volume.")

        # Remove the temporary file
        os.remove(temp_file_path)

        if create_progress_dialog:
            progress_dialog.close()

        return volume_node

    except requests.exceptions.RequestException as e:
        slicer.util.errorDisplay(f"Failed to download the NRRD file from:\n{url}\n\nError: {e}", windowTitle="Download Failed")
    except Exception as e:
        slicer.util.errorDisplay(f"Failed to process or load the NRRD file.\n\nError: {e}", windowTitle="Processing Failed")

    return None

class SinoReconsVisual2Logic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)

class SinoReconsVisual2Test(ScriptedLoadableModuleTest):
    def setup(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setup()
        self.delayDisplay("Test not implemented.")