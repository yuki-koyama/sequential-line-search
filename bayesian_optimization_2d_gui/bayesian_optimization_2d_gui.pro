QT += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = bayesian_optimization_2d_gui
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
    widgetsigma.cpp \
    widgetmu.cpp \
    widgetei.cpp \
    ../main/gaussianprocessregressor.cpp \
    ../main/expectedimprovementmaximizer.cpp \
    ../main/colorutility.cpp \
    ../main/utility.cpp \
    ../main/preferenceregressor.cpp \
    ../main/nloptutility.cpp \
    ../main/slicesampler.cpp \
    ../main/regressor.cpp

HEADERS += \
    mainwindow.h \
    mainwidget.h \
    core.h \
    widgetsigma.h \
    widgetmu.h \
    widgetei.h \
    ../main/gaussianprocessregressor.h \
    ../main/expectedimprovementmaximizer.h \
    ../main/colorutility.h \
    ../main/utility.h \
    ../main/preferenceregressor.h \
    ../main/nloptutility.h \
    ../main/regressor.h \
    ../main/slicesampler.h

FORMS += \
    mainwindow.ui
