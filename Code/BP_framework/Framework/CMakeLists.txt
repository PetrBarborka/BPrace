
cmake_minimum_required (VERSION 2.6)

project (Framework)

add_definitions(-std=c++11)

set( CMAKE_PREFIX_PATH "~/Qt5.5.1/5.5/clang_64/" )
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

find_package( OpenCV REQUIRED )
find_package( Qt5 REQUIRED Core Gui Widgets )
find_package( Boost COMPONENTS program_options system filesystem REQUIRED )

set ( contrib_LIBS  -L/usr/local/lib
        -lopencv_core
        -lopencv_highgui
        -lopencv_imgproc
        -lopencv_imgcodecs
        -lopencv_features2d
        -lopencv_xfeatures2d
        -lopencv_calib3d )

#set ( HEADERS lib/json/src/json.hpp include/utility.hpp include/detection.hpp include/description.hpp include/homography.hpp )

set ( SOURCES src/utility.cpp src/detection.cpp src/description.cpp src/homography.cpp)

include_directories( ${Boost_INCLUDE_DIR} )
include_directories( include )
include_directories( lib/json/src )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_library ( Framework STATIC ${SOURCES} )
target_link_libraries( Framework ${Boost_LIBRARIES} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY}  ${OpenCV_LIBS} ${contrib_LIBS} Qt5::Core)