#-----------------------------------------------------------------------------
set(MODULE_NAME SinoReconsVisual2)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  # FIXME: This file has been manually generated using rcc.exe from a local Qt installation...
  # > rcc.exe -binary .\3Dslicer\CT-Wood\CTWood\SinoReconsVisual2\Resources\sinoReconsVisual.qrc -o .\3Dslicer\CT-Wood\CTWood\SinoReconsVisual2\Resources\sinoReconsVisual.rcc
  # Ideally it would be generated as part of this CMake build setup but I'm unable to get it working properly...
  # See slicerFunctionAddPythonQtResources below.
  # - Julius Häger 2026-05-11
  Resources/sinoReconsVisual.rcc
  Resources/UI/${MODULE_NAME}.ui
  )

#-----------------------------------------------------------------------------
# FIXME: For some reason this doesn't work properly, and I do not understand why.
# - Julius Häger 2026-05-11
slicerFunctionAddPythonQtResources(
  Resources/sinoReconsVisual.qrc
  )

slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
