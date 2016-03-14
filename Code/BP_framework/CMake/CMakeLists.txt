cmake_minimum_required (VERSION 2.6)
project (BP)


#set (Qt5Widgets_DIR ~/Qt5.5.1/5.5/clang_64/lib/cmake/Qt5Widgets/)
set( CMAKE_PREFIX_PATH "~/Qt5.5.1/5.5/clang_64/" )

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

find_package( OpenCV REQUIRED ) 
find_package( Qt5 REQUIRED Core Gui Widgets )

if (Qt5_FOUND)
    message("Found Qt5")
    message("Includes: " ${Qt5_INCLUDE_DIRS})
    message("Libs: " ${Qt5_LIBS})
endif(Qt5_FOUND)

if(OpenCV_FOUND)
     message("Found OpenCV")
     message("Includes: " ${OpenCV_INCLUDE_DIRS})
     message("Libs: " ${OpenCV_LIBS})
endif(OpenCV_FOUND)

#set (OpenCV_DIR "/usr/local/lib/")

set ( contrib_LIBS  -L/usr/local/lib
                    -lopencv_core 
                    -lopencv_highgui 
                    -lopencv_imgproc 
                    -lopencv_imgcodecs 
                    -lopencv_features2d 
                    -lopencv_xfeatures2d 
                    -lopencv_calib3d )

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable (BP ../main.cpp)
target_link_libraries( BP ${OpenCV_LIBS} ${contrib_LIBS} Qt5::Core)
