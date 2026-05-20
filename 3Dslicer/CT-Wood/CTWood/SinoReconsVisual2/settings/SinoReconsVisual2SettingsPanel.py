import qt
import slicer
import ctk

from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

class SinoReconsVisual2SettingsPanel(qt.QObject):

    def __init__(self, settingsPanel: ctk.ctkSettingsPanel, mainWidget):
        super().__init__(settingsPanel)
        self.settingsPanel = settingsPanel
        self.ui = slicer.util.childWidgetVariables(settingsPanel)

        self.ui.sourceTrajectoryColorPickerButton.colorChanged.connect(mainWidget.sourceTrajectoryColorChanged)
        self.settingsPanel.registerProperty(
            "SinoReconsVisual2/TrajectoryColor",
            self.ui.sourceTrajectoryColorPickerButton,
            "color",
            # I couldn't find a SIGNAL macro equivalent in pythonqt - Julius Häger 2026-05-20
            b"2colorChanged(QColor)\0",
            _("Selected Color")
            )
        
        self.ui.sourceColorPickerButton.colorChanged.connect(mainWidget.sourceColorChanged)
        self.settingsPanel.registerProperty(
            "SinoReconsVisual2/SourceColor",
            self.ui.sourceColorPickerButton,
            "color",
            # I couldn't find a SIGNAL macro equivalent in pythonqt - Julius Häger 2026-05-20
            b"2colorChanged(QColor)\0",
            _("Source Color")
            )
        
        self.ui.detectorColorPickerButton.colorChanged.connect(mainWidget.detectorColorChanged)
        self.settingsPanel.registerProperty(
            "SinoReconsVisual2/DetectorColor",
            self.ui.detectorColorPickerButton,
            "color",
            # I couldn't find a SIGNAL macro equivalent in pythonqt - Julius Häger 2026-05-20
            b"2colorChanged(QColor)\0",
            _("Detector Color")
            )
        
        self.ui.fovRaysColorPickerButton.colorChanged.connect(mainWidget.fovRaysColorChanged)
        self.settingsPanel.registerProperty(
            "SinoReconsVisual2/FOVRayColor",
            self.ui.fovRaysColorPickerButton,
            "color",
            # I couldn't find a SIGNAL macro equivalent in pythonqt - Julius Häger 2026-05-20
            b"2colorChanged(QColor)\0",
            _("Field of View Ray Color")
            )
        
        self.ui.boundsColorPickerButton.colorChanged.connect(mainWidget.boundsColorChanged)
        self.settingsPanel.registerProperty(
            "SinoReconsVisual2/BoundsColor",
            self.ui.boundsColorPickerButton,
            "color",
            # I couldn't find a SIGNAL macro equivalent in pythonqt - Julius Häger 2026-05-20
            b"2colorChanged(QColor)\0",
            _("Bounds Color")
            )
        
        self.ui.sinogramOutlineColorPickerButton.colorChanged.connect(mainWidget.sinogramOutlineColorChanged)
        self.settingsPanel.registerProperty(
            "SinoReconsVisual2/SinogramOutlineColor",
            self.ui.sinogramOutlineColorPickerButton,
            "color",
            # I couldn't find a SIGNAL macro equivalent in pythonqt - Julius Häger 2026-05-20
            b"2colorChanged(QColor)\0",
            _("Sinogram Outline Color")
            )

        self.ui.selectedROIColorPickerButton.colorChanged.connect(mainWidget.selectedROIColorChanged)
        self.settingsPanel.registerProperty(
            "SinoReconsVisual2/ROI/SelectedColor",
            self.ui.selectedROIColorPickerButton,
            "color",
            # I couldn't find a SIGNAL macro equivalent in pythonqt - Julius Häger 2026-05-20
            b"2colorChanged(QColor)\0",
            _("Selected Color")
            )
        
        self.ui.inactiveROIColorPickerButton.colorChanged.connect(mainWidget.inactiveROIColorChanged)
        self.settingsPanel.registerProperty(
            "SinoReconsVisual2/ROI/InactiveColor",
            self.ui.inactiveROIColorPickerButton,
            "color",
            # I couldn't find a SIGNAL macro equivalent in pythonqt - Julius Häger 2026-05-20
            b"2colorChanged(QColor)\0",
            _("Inactive Color")
            )
        
        self.ui.roiSinogramRangeColorPickerButton.colorChanged.connect(mainWidget.roiSinogramRangeColorChanged)
        self.settingsPanel.registerProperty(
            "SinoReconsVisual2/ROI/SinogramRangeColor",
            self.ui.roiSinogramRangeColorPickerButton,
            "color",
            # I couldn't find a SIGNAL macro equivalent in pythonqt - Julius Häger 2026-05-20
            b"2colorChanged(QColor)\0",
            _("Sinogram Range Color")
            )
        
        self.settingsPanel.settingChanged.connect(lambda k, v: print(f"settings changed! {k} = {v}"))

