import qt
import slicer
import ctk

from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

class SinoReconsVisual2SettingsPanel(qt.QObject):

    @property
    def selectedROIColor(self) -> tuple[float,float,float]:
        return (1.0, 0.0, 0.0)

    @property
    def selectedROIOpacity(self) -> float:
        return 1.0
    
    @property
    def unselectedROIColor(self) -> tuple[float,float,float]:
        return (0.1, 0.1, 0.1)

    @property
    def unselectedROIOpacity(self) -> float:
        return 0.2

    def __init__(self, settingsPanel: ctk.ctkSettingsPanel):
        super().__init__(settingsPanel)
        self.settingsPanel = settingsPanel

        self.settingsPanel.registerProperty(
            "SinoReconsVisual2/ROI/SelectedColor",
            self,
            "selectedROIColor",
            b"selectedROIColorChanged(double,double,double)",
            _("Selected Color")
            )
        
    @qt.Slot(float,float,float)
    def selectedROIColorChanged(self,r,g,b):
        print("roi color", r, g, b)