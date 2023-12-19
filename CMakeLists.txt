cmake_minimum_required(VERSION 3.1)

project(CeresLearn LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 11)

find_package(Ceres REQUIRED)

include_directories(
  ${CERES_INCLUDE_DIRS}
)

list(APPEND OBJECTS_TO_LINK ${CERES_LIBRARIES})

set(TEST_LIST
  TestHelloWorld
  TestHelloWorldAnalytic
  TestPowell
  TestCurveFiting
)

foreach(TEST ${TEST_LIST})
  add_executable(${TEST} test/${TEST}.cc)
  target_link_libraries(${TEST} ${OBJECTS_TO_LINK})
endforeach(TEST ${TEST_LIST})

