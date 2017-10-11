QT += core gui concurrent

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = sequential_line_search_2d_gui
TEMPLATE = app

CONFIG += \
    c++11

QMAKE_CXXFLAGS = \
    -Wno-deprecated-register \
    -Wno-inconsistent-missing-override

macx {
QMAKE_LFLAGS += \
    -mmacosx-version-min=10.12 # QMAKE_MACOSX_DEPLOYMENT_TARGET does not work?
}

INCLUDEPATH += \
    $$PWD/../main \
    $$PWD/../include

LIBS += \
    $$PWD/../lib/libnlopt.a

SOURCES += \
    main.cpp \
    mainwindow.cpp \
    mainwidget.cpp \
    core.cpp \
    widgetpreview.cpp \
    ../main/gaussianprocessregressor.cpp \
    ../main/expectedimprovementmaximizer.cpp \
    ../main/colorutility.cpp \
    ../main/utility.cpp \
    ../main/preferenceregressor.cpp \
    ../main/nloptutility.cpp \
    ../main/sliderutility.cpp \
    ../main/slicesampler.cpp \
    ../main/slider.cpp \
    ../main/regressor.cpp

HEADERS += \
    mainwindow.h \
    mainwidget.h \
    core.h \
    widgetpreview.h \
    ../main/gaussianprocessregressor.h \
    ../main/expectedimprovementmaximizer.h \
    ../main/colorutility.h \
    ../main/utility.h \
    ../main/preferenceregressor.h \
    ../main/preference.h \
    ../main/nloptutility.h \
    ../main/sliderutility.h \
    ../main/slicesampler.h \
    ../main/slider.h

FORMS += \
    mainwindow.ui
