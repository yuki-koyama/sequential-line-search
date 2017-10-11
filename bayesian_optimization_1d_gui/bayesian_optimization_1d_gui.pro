QT += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = bayesian_optimization_1d_gui
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

SOURCES += main.cpp\
    mainwindow.cpp \
    mainwidget.cpp \
    core.cpp \
    ../main/gaussianprocessregressor.cpp \
    ../main/expectedimprovementmaximizer.cpp \
    ../main/nloptutility.cpp \
    ../main/preferenceregressor.cpp \
    ../main/utility.cpp \
    ../main/slicesampler.cpp \
    ../main/regressor.cpp

HEADERS  += mainwindow.h \
    mainwidget.h \
    core.h \
    ../main/gaussianprocessregressor.h \
    ../main/expectedimprovementmaximizer.h \
    ../main/nloptutility.h \
    ../main/preferenceregressor.h \
    ../main/utility.h \
    ../main/slicesampler.h \
    ../main/regressor.h

FORMS += \
    mainwindow.ui
