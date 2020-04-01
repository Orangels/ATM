cmake_minimum_required(VERSION 3.10)
project(ATM)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O0 -ggdb -Wall -fPIC -ggdb -O0") #-O0 -ggdb

find_package(CUDA)
find_package(OpenCV)
find_package(Protobuf REQUIRED)

set(cppCommonPath ${CMAKE_SOURCE_DIR}/src/cppCommon)

include_directories(${OpenCV_INCLUDE_DIRS} include core src/box_tracking
        /usr/include  ${PROTOBUF_INCLUDE_DIR}  ${cppCommonPath}/include
        src/TREngine
        /usr/local/ffmpeg/include
        src/projectUtils
        /usr/local/include
        )

add_subdirectory(${cppCommonPath}/src/Common)

if ("${CMAKE_SYSTEM}" MATCHES ".*tegra")
    set(cppComBranch tx2nano)
    set(trtLibPath /usr/lib/aarch64-linux-gnu)
    set(CUDA_CUBLAS_LIBRARIES /usr/local/cuda/targets/aarch64-linux/lib/stubs/libcublas.so)
    #    execute_process(COMMAND  ln -s /srv/models ${CMAKE_SOURCE_DIR}/../../../models)
else()
    set(cppComBranch ubuntu)
    set(trtLibPath /usr/lib/x86_64-linux-gnu)
    set(CUDA_NVCC_FLAGS
            ${CUDA_NVCC_FLAGS}
            -g -G  # -O3 #
            -gencode arch=compute_30,code=sm_30
            -gencode arch=compute_35,code=sm_35
            -gencode arch=compute_50,code=sm_50
            -gencode arch=compute_52,code=sm_52
            -gencode arch=compute_61,code=sm_61
            -gencode arch=compute_62,code=sm_62
            -gencode arch=compute_70,code=sm_70
            )
endif()
execute_process(COMMAND  git clone -b ${cppComBranch} git@172.16.105.143:WAR/cppcommon.git ${cppCommonPath})
execute_process(COMMAND  protoc  --cpp_out=${CMAKE_SOURCE_DIR}/src/TREngine  -I=${CMAKE_SOURCE_DIR}/src/TREngine ${CMAKE_SOURCE_DIR}/src/TREngine/caffe.proto)
execute_process(COMMAND  protoc  --cpp_out=${CMAKE_SOURCE_DIR}/src/TREngine  -I=${CMAKE_SOURCE_DIR}/src/TREngine ${CMAKE_SOURCE_DIR}/src/TREngine/postPara3DKeyPoints.proto)
execute_process(COMMAND  protoc  --cpp_out=${CMAKE_SOURCE_DIR}/src/TREngine  -I=${CMAKE_SOURCE_DIR}/src/TREngine ${CMAKE_SOURCE_DIR}/src/TREngine/onnx.proto)
execute_process(COMMAND  protoc  --cpp_out=${CMAKE_SOURCE_DIR}/src/TREngine  -I=${CMAKE_SOURCE_DIR}/src/TREngine ${CMAKE_SOURCE_DIR}/src/TREngine/faceFeatures.proto)
if(9 LESS ${CUDA_VERSION_MAJOR})
    execute_process(COMMAND rm ${CMAKE_SOURCE_DIR}/src/float32halfConvertor.cpp)
endif()
file(GLOB sources src/TREngine/*.cpp src/TREngine/*.cc onnx/*.cc src/projectUtils/*cpp)
file(GLOB cudaSources src/TREngine/*.cu)


set(CMAKE_CXX_STANDARD 11)

cuda_add_library(inferLib STATIC ${cudaSources})


add_executable(ATM tools/main.cpp include/utils/image_deliver.h src/utils/image_deliver.cpp include/collect.h
        src/collect.cpp include/engine/hf_det.h src/engine/hf_det.cpp include/structures/image.h
        src/structures/image.cpp include/tasks/solver.h src/tasks/solver.cpp include/tasks/hop_detect.h
        src/tasks/hop_detect.cpp core/config.h include/utils/misc.h src/utils/misc.cpp
        include/structures/structs.h include/structures/instance_group.h include/structures/instance.h
        src/structures/instance.cpp src/structures/instance_group.cpp include/engine/hand_det.h
        include/engine/hop_det.h src/engine/hand_det.cpp src/engine/hop_det.cpp src/box_tracking/box_tracking.cpp
        src/box_tracking/box_tracking.h src/box_tracking/Hungarian/Hungarian.cpp
        src/box_tracking/Hungarian/Hungarian.h include/utils/track.h src/utils/track.cpp
        include/utils/vis.h src/utils/vis.cpp include/tasks/group_detect.h src/tasks/group_detect.cpp
        include/tasks/entry_detect.h src/tasks/entry_detect.cpp include/tasks/hand_detect.h src/tasks/hand_detect.cpp
        include/tasks/turnround_detect.h src/tasks/turnround_detect.cpp include/structures/face_library.h
        src/structures/face_library.cpp
        ${sources})

target_link_libraries(ATM ${OpenCV_LIBS}
        ${PROTOBUF_LIBRARIES} inferLib nvcaffe_parser nvinfer nvinfer_plugin nvparsers  ${CUDA_CUBLAS_LIBRARIES} /usr/local/cuda/lib64/stubs/libcusolver.so ${CUDA_LIBRARIES} ${trtLibPath}/libcudnn.so
        #${cppCommonPath}/build/libCommon.so
        Common
        /usr/local/ffmpeg/lib/libavcodec.so
        /usr/local/ffmpeg/lib/libavdevice.so
        /usr/local/ffmpeg/lib/libavfilter.so
        /usr/local/ffmpeg/lib/libavformat.so
        /usr/local/ffmpeg/lib/libavutil.so
        /usr/local/ffmpeg/lib/libpostproc.so
        #        /usr/local/ffmpeg/lib/libswresample.so
        /usr/local/ffmpeg/lib/libswscale.so
        /usr/local/lib/libgearman.so
        )
