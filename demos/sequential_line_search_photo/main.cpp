#include "mainwindow.hpp"
#include <QApplication>
#include <QSurface>

int main(int argc, char* argv[])
{
    QApplication a(argc, argv);

    Q_INIT_RESOURCE(enhancer_resources);

#if defined(__APPLE__)
    QSurfaceFormat format;
    format.setVersion(3, 2);
    format.setProfile(QSurfaceFormat::CoreProfile);
    QSurfaceFormat::setDefaultFormat(format);
#endif

    MainWindow w;
    w.show();

    return a.exec();
}
