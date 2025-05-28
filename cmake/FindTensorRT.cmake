if(NOT TensorRT_ROOT)
    if(WIN32)
        set(TensorRT_ROOT "C:/Program Files/NVIDIA Corporation/TensorRT-10.9.0.34")
    else()
        set(TensorRT_ROOT "/usr")
    endif()
endif()

if(WIN32)
    set(TensorRT_INCLUDE_PATHS "${TensorRT_ROOT}/include")
    set(TensorRT_LIB_PATHS "${TensorRT_ROOT}/lib")
else()
    set(TensorRT_INCLUDE_PATHS "${TensorRT_ROOT}/include" "/usr/include/x86_64-linux-gnu")
    set(TensorRT_LIB_PATHS "${TensorRT_ROOT}/lib" "/usr/lib/x86_64-linux-gnu")
endif()

find_path(
    TensorRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS ${TensorRT_INCLUDE_PATHS}
)

function(_tensorrt_get_version)
    unset(TensorRT_VERSION_STRING PARENT_SCOPE)
    set(_hdr_file "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")

    if(NOT EXISTS "${_hdr_file}")
        return()
    endif()

    file(STRINGS "${_hdr_file}" VERSION_STRINGS REGEX "#define NV_TENSORRT_.*")

    if(NOT VERSION_STRINGS)
        return()
    endif()

    foreach(TYPE MAJOR MINOR PATCH BUILD)
        string(REGEX MATCH "NV_TENSORRT_${TYPE} [0-9]+" TRT_TYPE_STRING ${VERSION_STRINGS})
        if(TRT_TYPE_STRING)
            string(REGEX MATCH "[0-9]+" TensorRT_VERSION_${TYPE} ${TRT_TYPE_STRING})
        else()
            unset(TensorRT_VERSION_${TYPE})
        endif()
    endforeach()

    if(TensorRT_VERSION_MAJOR AND TensorRT_VERSION_MINOR AND TensorRT_VERSION_PATCH AND TensorRT_VERSION_BUILD)
        set(TensorRT_VERSION_STRING
            "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}.${TensorRT_VERSION_BUILD}"
            PARENT_SCOPE)
    endif()
endfunction()

_tensorrt_get_version()

if(WIN32)
    find_library(
        TensorRT_LIBRARY
        NAMES "nvinfer_${TensorRT_VERSION_MAJOR}" nvinfer_10
        PATHS ${TensorRT_LIB_PATHS}
    )
else()
    find_library(
        TensorRT_LIBRARY
        NAMES nvinfer
        PATHS ${TensorRT_LIB_PATHS}
    )
endif()

if(TensorRT_LIBRARY)
    set(TensorRT_LIBRARIES ${TensorRT_LIBRARIES} ${TensorRT_LIBRARY})
endif()

if(TensorRT_FIND_COMPONENTS)
    list(REMOVE_ITEM TensorRT_FIND_COMPONENTS "nvinfer")

    if("OnnxParser" IN_LIST TensorRT_FIND_COMPONENTS)
        find_path(
            TensorRT_OnnxParser_INCLUDE_DIR
            NAMES NvOnnxParser.h
            PATHS ${TensorRT_INCLUDE_PATHS}
        )

        if(WIN32)
            find_library(
                TensorRT_OnnxParser_LIBRARY
                NAMES "nvonnxparser_${TensorRT_VERSION_MAJOR}" nvonnxparser_10
                PATHS ${TensorRT_LIB_PATHS}
            )
        else()
            find_library(
                TensorRT_OnnxParser_LIBRARY
                NAMES nvonnxparser
                PATHS ${TensorRT_LIB_PATHS}
            )
        endif()

        if(TensorRT_OnnxParser_LIBRARY AND TensorRT_LIBRARIES)
            set(TensorRT_LIBRARIES ${TensorRT_LIBRARIES} ${TensorRT_OnnxParser_LIBRARY})
            set(TensorRT_OnnxParser_FOUND TRUE)
        endif()

        if(WIN32)
            find_file(
                TensorRT_OnnxParser_DLL
                NAMES "nvonnxparser_${TensorRT_VERSION_MAJOR}.dll" nvonnxparser_10.dll
                HINTS ${TensorRT_ROOT}
                PATH_SUFFIXES bin
            )
        endif()
    endif()

    if("Plugin" IN_LIST TensorRT_FIND_COMPONENTS)
        find_path(
            TensorRT_Plugin_INCLUDE_DIR
            NAMES NvInferPlugin.h
            PATHS ${TensorRT_INCLUDE_PATHS}
        )

        if(WIN32)
            find_library(
                TensorRT_Plugin_LIBRARY
                NAMES "nvinfer_plugin_${TensorRT_VERSION_MAJOR}" nvinfer_plugin_10
                PATHS ${TensorRT_LIB_PATHS}
            )
        else()
            find_library(
                TensorRT_Plugin_LIBRARY
                NAMES nvinfer_plugin
                PATHS ${TensorRT_LIB_PATHS}
            )
        endif()

        if(TensorRT_Plugin_LIBRARY AND TensorRT_LIBRARIES)
            set(TensorRT_LIBRARIES ${TensorRT_LIBRARIES} ${TensorRT_Plugin_LIBRARY})
            set(TensorRT_Plugin_FOUND TRUE)
        endif()

        if(WIN32)
            find_file(
                TensorRT_Plugin_DLL
                NAMES "nvinfer_plugin_${TensorRT_VERSION_MAJOR}.dll" nvinfer_plugin_10.dll
                HINTS ${TensorRT_ROOT}
                PATH_SUFFIXES bin
            )
        endif()
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    TensorRT
    FOUND_VAR TensorRT_FOUND
    REQUIRED_VARS TensorRT_LIBRARY TensorRT_LIBRARIES TensorRT_INCLUDE_DIR
    VERSION_VAR TensorRT_VERSION_STRING
    HANDLE_COMPONENTS
)

if(NOT TARGET TensorRT::NvInfer)
    add_library(TensorRT::NvInfer SHARED IMPORTED)
    target_include_directories(TensorRT::NvInfer SYSTEM INTERFACE "${TensorRT_INCLUDE_DIR}")
    if(WIN32)
        set_property(TARGET TensorRT::NvInfer PROPERTY IMPORTED_LOCATION "${TensorRT_DLL}")
        set_property(TARGET TensorRT::NvInfer PROPERTY IMPORTED_IMPLIB "${TensorRT_LIBRARY}")
    else()
        set_property(TARGET TensorRT::NvInfer PROPERTY IMPORTED_LOCATION "${TensorRT_LIBRARY}")
    endif()
endif()

if(NOT TARGET TensorRT::OnnxParser AND "OnnxParser" IN_LIST TensorRT_FIND_COMPONENTS)
    add_library(TensorRT::OnnxParser SHARED IMPORTED)
    target_include_directories(TensorRT::OnnxParser SYSTEM INTERFACE "${TensorRT_OnnxParser_INCLUDE_DIR}")
    target_link_libraries(TensorRT::OnnxParser INTERFACE TensorRT::NvInfer)
    if(WIN32)
        set_property(TARGET TensorRT::OnnxParser PROPERTY IMPORTED_LOCATION "${TensorRT_OnnxParser_DLL}")
        set_property(TARGET TensorRT::OnnxParser PROPERTY IMPORTED_IMPLIB "${TensorRT_OnnxParser_LIBRARY}")
    else()
        set_property(TARGET TensorRT::OnnxParser PROPERTY IMPORTED_LOCATION "${TensorRT_OnnxParser_LIBRARY}")
    endif()
endif()

if(NOT TARGET TensorRT::Plugin AND "Plugin" IN_LIST TensorRT_FIND_COMPONENTS)
    add_library(TensorRT::Plugin SHARED IMPORTED)
    target_include_directories(TensorRT::Plugin SYSTEM INTERFACE "${TensorRT_Plugin_INCLUDE_DIR}")
    target_link_libraries(TensorRT::Plugin INTERFACE TensorRT::NvInfer)
    if(WIN32)
        set_property(TARGET TensorRT::Plugin PROPERTY IMPORTED_LOCATION "${TensorRT_Plugin_DLL}")
        set_property(TARGET TensorRT::Plugin PROPERTY IMPORTED_IMPLIB "${TensorRT_Plugin_LIBRARY}")
    else()
        set_property(TARGET TensorRT::Plugin PROPERTY IMPORTED_LOCATION "${TensorRT_Plugin_LIBRARY}")
    endif()
endif()

mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_LIBRARY TensorRT_LIBRARIES)