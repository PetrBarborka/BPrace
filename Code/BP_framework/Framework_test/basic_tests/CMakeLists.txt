
include_directories( ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

message( "GTEST SOURCE DIR: ")
message( ${gtest_SOURCE_DIR} )

add_executable(runTests test.cpp)

target_link_libraries( runTests gtest gtest_main )
target_link_libraries( runTests Framework )