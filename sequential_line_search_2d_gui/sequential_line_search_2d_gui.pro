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
    $$PWD/../common \
    $$PWD/../include

LIBS += \
    $$PWD/../lib/libnlopt.a

SOURCES += \
    main.cpp \
    mainwindow.cpp \
    mainwidget.cpp \
    core.cpp \
    widgetpreview.cpp \
    ../common/gaussianprocessregressor.cpp \
    ../common/expectedimprovementmaximizer.cpp \
    ../common/colorutility.cpp \
    ../common/utility.cpp \
    ../common/preferenceregressor.cpp \
    ../common/nloptutility.cpp \
    ../common/sliderutility.cpp \
    ../common/slicesampler.cpp \
    ../common/slider.cpp \
    ../common/regressor.cpp

HEADERS += \
    mainwindow.h \
    mainwidget.h \
    core.h \
    widgetpreview.h \
    ../common/gaussianprocessregressor.h \
    ../common/expectedimprovementmaximizer.h \
    ../common/colorutility.h \
    ../common/utility.h \
    ../common/preferenceregressor.h \
    ../common/preference.h \
    ../common/nloptutility.h \
    ../common/sliderutility.h \
    ../common/slicesampler.h \
    ../common/slider.h

FORMS += \
    mainwindow.ui
