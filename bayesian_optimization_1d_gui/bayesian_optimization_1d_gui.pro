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
    $$PWD/../common \
    $$PWD/../include

LIBS += \
    $$PWD/../lib/libnlopt.a

SOURCES += main.cpp\
    mainwindow.cpp \
    mainwidget.cpp \
    core.cpp \
    ../common/gaussianprocessregressor.cpp \
    ../common/expectedimprovementmaximizer.cpp \
    ../common/nloptutility.cpp \
    ../common/preferenceregressor.cpp \
    ../common/utility.cpp \
    ../common/slicesampler.cpp \
    ../common/regressor.cpp

HEADERS  += mainwindow.h \
    mainwidget.h \
    core.h \
    ../common/gaussianprocessregressor.h \
    ../common/expectedimprovementmaximizer.h \
    ../common/nloptutility.h \
    ../common/preferenceregressor.h \
    ../common/utility.h \
    ../common/slicesampler.h \
    ../common/regressor.h

FORMS    += mainwindow.ui
