import logging
import os
import math
from typing import Optional

import vtk
import qt

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

from slicer import vtkMRMLScalarVolumeNode

# Define the main module class
class LoadCTData(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("LoadCTData")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Utilities")]
        self.parent.dependencies = []
        self.parent.contributors = ["Oualid Burström (LTU)"]
        self.parent.helpText = _("""
        The LoadCTData extension enables users to upload NRRD files and perform manipulations such as rotations around the X, Y, and Z axes, as well as translations along these axes.
        """)
        self.parent.acknowledgementText = _(""" """)
        # Set the icon for the module
        iconsPath = os.path.join(os.path.dirname(__file__), "Resources", "Icons")
        iconPath = os.path.join(iconsPath, "LoadCTData2.png")
        if os.path.exists(iconPath):
            self.parent.icon = qt.QIcon(iconPath)

# Define the widget class for the module
class LoadCTDataWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None

        # Initialize rotation and translation values
        self.rotationValues = {'x': 0, 'y': 0, 'z': 0}
        self.translationValues = {'x': 0, 'y': 0, 'z': 0}

    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)
        # Load the UI from .ui file
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/LoadCTData.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        self.logic = LoadCTDataLogic()

        # Set up event observers
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Connect UI elements to methods
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.LoadCTDataButton.connect('clicked(bool)', self.onLoadCTDataButton)
        self.ui.rotateXLeftButton.connect('clicked(bool)', lambda: self.onRotateButton('x', opposite=True))
        self.ui.rotateXRightButton.connect('clicked(bool)', lambda: self.onRotateButton('x'))
        self.ui.rotateYLeftButton.connect('clicked(bool)', lambda: self.onRotateButton('y', opposite=True))
        self.ui.rotateYRightButton.connect('clicked(bool)', lambda: self.onRotateButton('y'))
        self.ui.rotateZLeftButton.connect('clicked(bool)', lambda: self.onRotateButton('z', opposite=True))
        self.ui.rotateZRightButton.connect('clicked(bool)', lambda: self.onRotateButton('z'))
        self.ui.translateXLeftButton.connect('clicked(bool)', lambda: self.onTranslateButton('x', opposite=True))
        self.ui.translateXRightButton.connect('clicked(bool)', lambda: self.onTranslateButton('x'))
        self.ui.translateYLeftButton.connect('clicked(bool)', lambda: self.onTranslateButton('y', opposite=True))
        self.ui.translateYRightButton.connect('clicked(bool)', lambda: self.onTranslateButton('y'))
        self.ui.translateZLeftButton.connect('clicked(bool)', lambda: self.onTranslateButton('z', opposite=True))
        self.ui.translateZRightButton.connect('clicked(bool)', lambda: self.onTranslateButton('z'))

        # Set MRML Scene for selectors
        self.ui.inputVolumeSelector.setMRMLScene(slicer.mrmlScene)
        self.ui.outputVolumeSelector.setMRMLScene(slicer.mrmlScene)

    def cleanup(self) -> None:
        self.removeObservers()

    def enter(self) -> None:
        pass

    def exit(self) -> None:
        pass

    def onSceneStartClose(self, caller, event) -> None:
        pass

    def onSceneEndClose(self, caller, event) -> None:
        pass

    def onApplyButton(self) -> None:
        # Handle apply button click
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            self.logic.process(self.ui.inputVolumeSelector.currentNode(), self.ui.outputVolumeSelector.currentNode(),
                               self.ui.imageThresholdSliderWidget.value, self.ui.invertThresholdCheckBox.checked)
            if self.ui.invertedVolumeSelector.currentNode():
                self.logic.process(self.ui.inputVolumeSelector.currentNode(), self.ui.invertedVolumeSelector.currentNode(),
                                   self.ui.imageThresholdSliderWidget.value, not self.ui.invertThresholdCheckBox.checked, showResult=False)

    def onLoadCTDataButton(self) -> None:
        # Handle load CT data button click
        fileDialog = qt.QFileDialog(self.parent)
        fileDialog.setNameFilter("Data Files (*.nrrd *.nii *.nii.gz)")
        fileDialog.setFileMode(qt.QFileDialog.ExistingFile)
        if fileDialog.exec_():
            selectedFile = fileDialog.selectedFiles()[0]
            logging.info(f"Selected file: {selectedFile}")
            loadedVolume = slicer.util.loadVolume(selectedFile)
            print(f"loadedVolume: {loadedVolume}")  # Print statement to check the value of loadedVolume
            if loadedVolume:
                logging.info(f"Loaded volume: {loadedVolume.GetName()}")
                print(f"Class of loadedVolume: {loadedVolume.GetClassName()}")  # Print the class of loadedVolume
                if slicer.mrmlScene.GetNodesByClass('vtkMRMLScalarVolumeNode').GetNumberOfItems() == 0:
                    logging.error("No volume nodes in the MRML scene.")
                else:
                    logging.info("Volume nodes found in the MRML scene.")
                
                # Set loaded volume to slice viewer and fit to window
                slicer.util.setSliceViewerLayers(background=loadedVolume)
                slicer.app.applicationLogic().FitSliceToAll()
                slicer.util.resetThreeDViews()

                # Explicitly check and set the current node
                logging.info(f"Before setting input volume selector, currentNode: {self.ui.inputVolumeSelector.currentNode()}")
                self.ui.inputVolumeSelector.setMRMLScene(slicer.mrmlScene)
                self.ui.inputVolumeSelector.setCurrentNode(loadedVolume)
                inputNode = self.ui.inputVolumeSelector.currentNode()
                logging.info(f"After setting input volume selector, inputNode: {inputNode}")
                if inputNode:
                    logging.info(f"Set input volume selector to: {inputNode.GetName()}")
                else:
                    logging.error("Failed to set input volume selector.")

                logging.info(f"Before setting output volume selector, currentNode: {self.ui.outputVolumeSelector.currentNode()}")
                self.ui.outputVolumeSelector.setMRMLScene(slicer.mrmlScene)
                self.ui.outputVolumeSelector.setCurrentNode(loadedVolume)
                outputNode = self.ui.outputVolumeSelector.currentNode()
                logging.info(f"After setting output volume selector, outputNode: {outputNode}")
                if outputNode:
                    logging.info(f"Set output volume selector to: {outputNode.GetName()}")
                else:
                    logging.error("Failed to set output volume selector.")
            else:
                logging.error("Failed to load volume.")
        else:
            logging.info("File dialog canceled.")

    def onRotateButton(self, axis: str, opposite: bool = False) -> None:
        # Handle rotation button click
        inputVolume = self.ui.inputVolumeSelector.currentNode()
        logging.info(f"inputVolume: {inputVolume}")
        if not inputVolume:
            slicer.util.errorDisplay("No input volume selected.")
            logging.error("No input volume selected.")
            return
        logging.info(f"Input volume: {inputVolume.GetName()}")
        angle = self.ui.rotationAngleSpinBox.value
        if opposite:
            angle = -angle
        self.rotationValues[axis] += angle
        logging.info(f"Rotating around {axis}-axis by {angle} degrees")
        self.logic.rotateVolume(inputVolume, axis, angle)
        self.updateRotationLabels()

    def onTranslateButton(self, axis: str, opposite: bool = False) -> None:
        # Handle translation button click
        inputVolume = self.ui.inputVolumeSelector.currentNode()
        logging.info(f"inputVolume: {inputVolume}")
        if not inputVolume:
            slicer.util.errorDisplay("No input volume selected.")
            logging.error("No input volume selected.")
            return
        logging.info(f"Input volume: {inputVolume.GetName()}")
        distance = self.ui.translationDistanceSpinBox.value
        if opposite:
            distance = -distance
        self.translationValues[axis] += distance
        logging.info(f"Translating along {axis}-axis by {distance} mm")
        self.logic.translateVolume(inputVolume, axis, distance)
        self.updateTranslationLabels()

    def updateRotationLabels(self) -> None:
        # Update rotation labels in the UI
        self.ui.rotateXValueLabel.setText(f"{self.rotationValues['x']}°")
        self.ui.rotateYValueLabel.setText(f"{self.rotationValues['y']}°")
        self.ui.rotateZValueLabel.setText(f"{self.rotationValues['z']}°")

    def updateTranslationLabels(self) -> None:
        # Update translation labels in the UI
        self.ui.translateXValueLabel.setText(f"{self.translationValues['x']} mm")
        self.ui.translateYValueLabel.setText(f"{self.translationValues['y']} mm")
        self.ui.translateZValueLabel.setText(f"{self.translationValues['z']} mm")

# Define the logic class for the module
class LoadCTDataLogic(ScriptedLoadableModuleLogic):
    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLScalarVolumeNode,
                imageThreshold: float,
                invert: bool = False,
                showResult: bool = True) -> None:
        # Process the volumes with thresholding
        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")
        import time
        startTime = time.time()
        logging.info("Processing started")
        cliParams = {
            "InputVolume": inputVolume.GetID(),
            "OutputVolume": outputVolume.GetID(),
            "ThresholdValue": imageThreshold,
            "ThresholdType": "Above" if invert else "Below",
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        slicer.mrmlScene.RemoveNode(cliNode)
        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")

    def rotateVolume(self, volumeNode: vtkMRMLScalarVolumeNode, axis: str, angle: float) -> None:
        # Rotate the volume around a specified axis
        if not volumeNode:
            raise ValueError("Volume node is invalid")
        transformNode = slicer.vtkMRMLLinearTransformNode()
        slicer.mrmlScene.AddNode(transformNode)
        matrix = vtk.vtkMatrix4x4()
        transformNode.GetMatrixTransformToParent(matrix)
        radians = math.radians(angle)
        if axis == 'x':
            matrix.SetElement(1, 1, math.cos(radians))
            matrix.SetElement(1, 2, -math.sin(radians))
            matrix.SetElement(2, 1, math.sin(radians))
            matrix.SetElement(2, 2, math.cos(radians))
        elif axis == 'y':
            matrix.SetElement(0, 0, math.cos(radians))
            matrix.SetElement(0, 2, math.sin(radians))
            matrix.SetElement(2, 0, -math.sin(radians))
            matrix.SetElement(2, 2, math.cos(radians))
        elif axis == 'z':
            matrix.SetElement(0, 0, math.cos(radians))
            matrix.SetElement(0, 1, -math.sin(radians))
            matrix.SetElement(1, 0, math.sin(radians))
            matrix.SetElement(1, 1, math.cos(radians))
        transformNode.SetMatrixTransformToParent(matrix)
        volumeNode.SetAndObserveTransformNodeID(transformNode.GetID())
        slicer.vtkSlicerTransformLogic().hardenTransform(volumeNode)
        slicer.mrmlScene.RemoveNode(transformNode)

    def translateVolume(self, volumeNode: vtkMRMLScalarVolumeNode, axis: str, distance: float) -> None:
        # Translate the volume along a specified axis
        if not volumeNode:
            raise ValueError("Volume node is invalid")
        transformNode = slicer.vtkMRMLLinearTransformNode()
        slicer.mrmlScene.AddNode(transformNode)
        matrix = vtk.vtkMatrix4x4()
        transformNode.GetMatrixTransformToParent(matrix)
        if axis == 'x':
            matrix.SetElement(0, 3, distance)
        elif axis == 'y':
            matrix.SetElement(1, 3, distance)
        elif axis == 'z':
            matrix.SetElement(2, 3, distance)
        transformNode.SetMatrixTransformToParent(matrix)
        volumeNode.SetAndObserveTransformNodeID(transformNode.GetID())
        slicer.vtkSlicerTransformLogic().hardenTransform(volumeNode)
        slicer.mrmlScene.RemoveNode(transformNode)

# Define the test class for the module
class LoadCTDataTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_LoadCTData1()

    def test_LoadCTData1(self):
        self.delayDisplay("Starting the test")
        import SampleData
        inputVolume = SampleData.downloadSample("LoadCTData1")
        self.delayDisplay("Loaded test data set")
        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)
        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100
        logic = LoadCTDataLogic()
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])
        self.delayDisplay("Test passed")
