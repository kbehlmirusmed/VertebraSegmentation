import os
import vtk, qt, ctk, slicer
import logging
from SegmentEditorEffects import *
import vtkITK
import SimpleITK as sitk
import sitkUtils
import math
import vtkSegmentationCorePython as vtkSegmentationCore 
import vtkSlicerSegmentationsModuleLogicPython as vtkSlicerSegmentationsModuleLogic
import SampleData

class VertebraSegmentation()

  # Load master volume
  sampleDataLogic = SampleData.SampleDataLogic()
  masterVolumeNode = sampleDataLogic.downloadCTACardio()

  ##gets the node coordinates to run the grow cut from later
  hierarchy = slicer.vtkMMRLSubjectHierarchyNode(slicer.mmrlScene)
  sceneItemID = hierarchy.GetSceneItemID()
  subjectItemID = hierarchy.GetItemChildWithName(sceneItemID,'Fiducial Nodes')
  fidList = slicer.util.getNode('FiducialNodes')


  # Create segmentation
  segmentationNode = slicer.vtkMRMLSegmentationNode()
  slicer.mrmlScene.AddNode(segmentationNode)
  segmentationNode.CreateDefaultDisplayNodes() # only needed for display
  segmentationNode.name = 'Lumbar'
  segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(masterVolumeNode)

  # Create seed segment inside lumbar and name
  lumbarSeed = fidList[0]
  lumbarSeed.SetRadius(3)
  lumbarSeed.Update()
  segmentationNode.AddSegmentFromClosedSurfaceRepresentation(lumbarSeed.GetOutput(), "Lumbar Vertebra", [1.0,0.0,0.0])


  def __init__(self, scriptedEffect):
    SegmentEditorThresholdEffect.__init__(self, scriptedEffect)
    scriptedEffect.name = 'Local Threshold'
    self.previewSteps = 4

  def updateMRMLFromGUI(self): ## sets the parameters for local threshold - grow cut at 6mm for feature size and 265.00 to 1009.00 for threshold range
    SegmentEditorThresholdEffect.updateMRMLFromGUI(self)
    featureSizeMm = 6.000 ##self.minimumMinimumFeatureSize.value
    self.scriptedEffect.setParameter(MINIMUM_FEATURE_MM_PARAMETER_NAME, featureSizeMm)

    segmentationAlgorithm = "GrowCut" ##self.segmentationAlgorithmSelector.currentText
    self.scriptedEffect.setParameter(SEGMENTATION_ALGORITHM_PARAMETER_NAME, segmentationAlgorithm)


  def runGrowCut(self, masterImageData, seedLabelmap, outputLabelmap): ## runs the grow cut - local threshold on the segment

    self.clippedMaskImageData = slicer.vtkOrientedImageData()
    intensityBasedMasking = self.scriptedEffect.parameterSetNode().GetMasterVolumeIntensityMask()
    segmentationNode = self.scriptedEffect.parameterSetNode().GetSegmentationNode()
    success = segmentationNode.GenerateEditMask(self.clippedMaskImageData,
      self.scriptedEffect.parameterSetNode().GetMaskMode(),
      masterImageData, # reference geometry
      "", # edited segment ID
      self.scriptedEffect.parameterSetNode().GetMaskSegmentID() if self.scriptedEffect.parameterSetNode().GetMaskSegmentID() else "",
      masterImageData if intensityBasedMasking else None,
      self.scriptedEffect.parameterSetNode().GetMasterVolumeIntensityMaskRange() if intensityBasedMasking else None)

    import vtkSlicerSegmentationsModuleLogicPython as vtkSlicerSegmentationsModuleLogic
    self.growCutFilter = vtkSlicerSegmentationsModuleLogic.vtkImageGrowCutSegment()
    self.growCutFilter.SetIntensityVolume(masterImageData)
    self.growCutFilter.SetSeedLabelVolume(seedLabelmap)
    self.growCutFilter.SetMaskVolume(self.clippedMaskImageData)
    self.growCutFilter.Update()
    outputLabelmap.ShallowCopy(self.growCutFilter.GetOutput())

  
  def apply(self, ijkPoints):
    kernelSizePixel = self.getKernelSizePixel()
    if kernelSizePixel[0]<=0 and kernelSizePixel[1]<=0 and kernelSizePixel[2]<=0:
      return

    qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)

    # Get parameter set node
    parameterSetNode = self.scriptedEffect.parameterSetNode()

    # Get parameters
    minimumThreshold = self.scriptedEffect.doubleParameter("MinimumThreshold")
    maximumThreshold = self.scriptedEffect.doubleParameter("MaximumThreshold")

    # Get modifier labelmap
    modifierLabelmap = self.scriptedEffect.defaultModifierLabelmap()

    # Get master volume image data
    masterImageData = self.scriptedEffect.masterVolumeImageData()

    # Set intensity range
    oldMasterVolumeIntensityMask = parameterSetNode.GetMasterVolumeIntensityMask()
    parameterSetNode.MasterVolumeIntensityMaskOn()
    oldIntensityMaskRange = parameterSetNode.GetMasterVolumeIntensityMaskRange()
    intensityRange = [265.00, 1009.00]
    if oldMasterVolumeIntensityMask:
      intensityRange = [max(oldIntensityMaskRange[0], minimumThreshold), min(oldIntensityMaskRange[1], maximumThreshold)]
    parameterSetNode.SetMasterVolumeIntensityMaskRange(intensityRange)

    roiNode = lumbarSeed ##self.roiSelector.currentNode()
    clippedMasterImageData = masterImageData
    if roiNode is not None:
      worldToImageMatrix = vtk.vtkMatrix4x4()
      masterImageData.GetWorldToImageMatrix(worldToImageMatrix)

      bounds = [0,0,0,0,0,0]
      roiNode.GetRASBounds(bounds)
      corner1RAS = [bounds[0], bounds[2], bounds[4], 1]
      corner1IJK = [0, 0, 0, 0]
      worldToImageMatrix.MultiplyPoint(corner1RAS, corner1IJK)

      corner2RAS = [bounds[1], bounds[3], bounds[5], 1]
      corner2IJK = [0, 0, 0, 0]
      worldToImageMatrix.MultiplyPoint(corner2RAS, corner2IJK)

      extent = [0, -1, 0, -1, 0, -1]
      for i in range(3):
          lowerPoint = min(corner1IJK[i], corner2IJK[i])
          upperPoint = max(corner1IJK[i], corner2IJK[i])
          extent[2*i] = int(math.floor(lowerPoint))
          extent[2*i+1] = int(math.ceil(upperPoint))

      imageToWorldMatrix = vtk.vtkMatrix4x4()
      masterImageData.GetImageToWorldMatrix(imageToWorldMatrix)
      clippedMasterImageData = slicer.vtkOrientedImageData()
      self.padder = vtk.vtkImageConstantPad()
      self.padder.SetInputData(masterImageData)
      self.padder.SetOutputWholeExtent(extent)
      self.padder.Update()
      clippedMasterImageData.ShallowCopy(self.padder.GetOutput())
      clippedMasterImageData.SetImageToWorldMatrix(imageToWorldMatrix)

    # Pipeline
    self.thresh = vtk.vtkImageThreshold()
    self.thresh.SetInValue(LABEL_VALUE)
    self.thresh.SetOutValue(BACKGROUND_VALUE)
    self.thresh.SetInputData(clippedMasterImageData)
    self.thresh.ThresholdBetween(minimumThreshold, maximumThreshold)
    self.thresh.SetOutputScalarTypeToUnsignedChar()
    self.thresh.Update()

    self.erode = vtk.vtkImageDilateErode3D()
    self.erode.SetInputConnection(self.thresh.GetOutputPort())
    self.erode.SetDilateValue(BACKGROUND_VALUE)
    self.erode.SetErodeValue(LABEL_VALUE)
    self.erode.SetKernelSize(
      kernelSizePixel[0],
      kernelSizePixel[1],
      kernelSizePixel[2])

    self.erodeCast = vtk.vtkImageCast()
    self.erodeCast.SetInputConnection(self.erode.GetOutputPort())
    self.erodeCast.SetOutputScalarTypeToUnsignedInt()
    self.erodeCast.Update()

    # Remove small islands
    self.islandMath = vtkITK.vtkITKIslandMath()
    self.islandMath.SetInputConnection(self.erodeCast.GetOutputPort())
    self.islandMath.SetFullyConnected(False)
    self.islandMath.SetMinimumSize(125)  # remove regions smaller than 5x5x5 voxels

    self.islandThreshold = vtk.vtkImageThreshold()
    self.islandThreshold.SetInputConnection(self.islandMath.GetOutputPort())
    self.islandThreshold.ThresholdByLower(BACKGROUND_VALUE)
    self.islandThreshold.SetInValue(BACKGROUND_VALUE)
    self.islandThreshold.SetOutValue(LABEL_VALUE)
    self.islandThreshold.SetOutputScalarTypeToUnsignedChar()
    self.islandThreshold.Update()

    # Points may be outside the region after it is eroded.
    # Snap the points to LABEL_VALUE voxels,
    snappedIJKPoints = self.snapIJKPointsToLabel(ijkPoints, self.islandThreshold.GetOutput())
    if snappedIJKPoints.GetNumberOfPoints() == 0:
      qt.QApplication.restoreOverrideCursor()
      return

    # Convert points to real data coordinates. Required for vtkImageThresholdConnectivity.
    seedPoints = vtk.vtkPoints()
    origin = masterImageData.GetOrigin()
    spacing = masterImageData.GetSpacing()
    for i in range(snappedIJKPoints.GetNumberOfPoints()):
      ijkPoint = snappedIJKPoints.GetPoint(i)
      seedPoints.InsertNextPoint(
        origin[0]+ijkPoint[0]*spacing[0],
        origin[1]+ijkPoint[1]*spacing[1],
        origin[2]+ijkPoint[2]*spacing[2])

    segmentationAlgorithm = self.scriptedEffect.parameter(SEGMENTATION_ALGORITHM_PARAMETER_NAME)
    if segmentationAlgorithm == SEGMENTATION_ALGORITHM_MASKING:
      self.runMasking(seedPoints, self.islandThreshold.GetOutput(), modifierLabelmap)

    else:
      self.floodFillingFilterIsland = vtk.vtkImageThresholdConnectivity()
      self.floodFillingFilterIsland.SetInputConnection(self.islandThreshold.GetOutputPort())
      self.floodFillingFilterIsland.SetInValue(SELECTED_ISLAND_VALUE)
      self.floodFillingFilterIsland.ReplaceInOn()
      self.floodFillingFilterIsland.ReplaceOutOff()
      self.floodFillingFilterIsland.ThresholdBetween(LABEL_VALUE, LABEL_VALUE)
      self.floodFillingFilterIsland.SetSeedPoints(seedPoints)
      self.floodFillingFilterIsland.Update()

      self.maskCast = vtk.vtkImageCast()
      self.maskCast.SetInputData(self.thresh.GetOutput())
      self.maskCast.SetOutputScalarTypeToUnsignedChar()
      self.maskCast.Update()

      self.imageMask = vtk.vtkImageMask()
      self.imageMask.SetInputConnection(self.floodFillingFilterIsland.GetOutputPort())
      self.imageMask.SetMaskedOutputValue(OUTSIDE_THRESHOLD_VALUE)
      self.imageMask.SetMaskInputData(self.maskCast.GetOutput())
      self.imageMask.Update()

      imageMaskOutput = slicer.vtkOrientedImageData()
      imageMaskOutput.ShallowCopy(self.imageMask.GetOutput())
      imageMaskOutput.CopyDirections(clippedMasterImageData)

      imageToWorldMatrix = vtk.vtkMatrix4x4()
      imageMaskOutput.GetImageToWorldMatrix(imageToWorldMatrix)

      segmentOutputLabelmap = slicer.vtkOrientedImageData()
      if segmentationAlgorithm == SEGMENTATION_ALGORITHM_GROWCUT:
        self.runGrowCut(clippedMasterImageData, imageMaskOutput, segmentOutputLabelmap)
      elif segmentationAlgorithm == SEGMENTATION_ALGORITHM_WATERSHED:
        self.runWatershed(clippedMasterImageData, imageMaskOutput, segmentOutputLabelmap)
      else:
        logging.error("Unknown segmentation algorithm: \"" + segmentationAlgorithm + "\"")

      segmentOutputLabelmap.SetImageToWorldMatrix(imageToWorldMatrix)

      self.selectedSegmentThreshold = vtk.vtkImageThreshold()
      self.selectedSegmentThreshold.SetInputData(segmentOutputLabelmap)
      self.selectedSegmentThreshold.ThresholdBetween(SELECTED_ISLAND_VALUE, SELECTED_ISLAND_VALUE)
      self.selectedSegmentThreshold.SetInValue(LABEL_VALUE)
      self.selectedSegmentThreshold.SetOutValue(BACKGROUND_VALUE)
      self.selectedSegmentThreshold.SetOutputScalarType(modifierLabelmap.GetScalarType())
      self.selectedSegmentThreshold.Update()
      modifierLabelmap.ShallowCopy(self.selectedSegmentThreshold.GetOutput())

    self.scriptedEffect.saveStateForUndo()
    self.scriptedEffect.modifySelectedSegmentByLabelmap(modifierLabelmap, slicer.qSlicerSegmentEditorAbstractEffect.ModificationModeAdd)

    parameterSetNode.SetMasterVolumeIntensityMask(oldMasterVolumeIntensityMask)
    parameterSetNode.SetMasterVolumeIntensityMaskRange(oldIntensityMaskRange)

    qt.QApplication.restoreOverrideCursor()
