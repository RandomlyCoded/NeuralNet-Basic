cmake_minimum_required(VERSION 3.14)

project(NeuralNet LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 REQUIRED COMPONENTS Core)
find_package(Qt6 REQUIRED COMPONENTS Core Quick Widgets)

set(NEURALNET_SRC
    NeuralNet_global.h

    advancednet.cpp advancednet.h
    advancednetviewprovider.h advancednetviewprovider.cpp

    neuralnet.cpp neuralnet.h
    neuralnetviewprovider.h neuralnetviewprovider.cpp
)

add_library(NeuralNet STATIC
    ${NEURALNET_SRC}
)

add_executable(NeuralNetView
    backend.h backend.cpp
    qmlnetviewconverter.h qmlnetviewconverter.cpp

    Application.h

    main.cpp

    main.qrc
)

add_executable(NeuralNetTests
    tests-main.cpp

    tests-main.qrc
    snake.h snake.cpp
    tests-playground.h tests-playground.cpp
)

add_executable(NeuralNet_Editor
    Editor-mainwindow.h Editor-mainwindow.cpp
    neuroneditor.h neuroneditor.cpp
    connectioneditor.h connectioneditor.cpp
    valueslider.h valueslider.cpp
    Editor.cpp Editor.h
    Editor-main.cpp
)

target_link_libraries(NeuralNet PRIVATE Qt6::Core Qt6::Quick)

target_compile_definitions(NeuralNet PRIVATE NeuralNet_LIBRARY)

target_compile_definitions(NeuralNetView
    PRIVATE $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:QT_QML_DEBUG>)
target_link_libraries(NeuralNetView
    PRIVATE Qt6::Quick NeuralNet)
target_link_libraries(NeuralNetTests
    PRIVATE Qt6::Quick NeuralNet)
target_link_libraries(NeuralNet_Editor
    PRIVATE Qt6::Quick Qt6::Widgets NeuralNet)
