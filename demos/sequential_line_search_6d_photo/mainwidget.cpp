#include "mainwidget.h"
#include <iostream>
#include <QPainter>
#include <QPaintEvent>
#include "core.h"
#include "mainwindow.h"
#include "gaussianprocessregressor.h"
#include "expectedimprovementmaximizer.h"

using Eigen::VectorXd;

namespace
{
Core& core = Core::getInstance();
}

MainWidget::MainWidget(QWidget *parent) :
    QWidget(parent)
{
    setAutoFillBackground(true);
}

void MainWidget::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    const QRect& rect = event->rect();
    const int    w    = rect.width();
    const int    h    = rect.height();

    // Draw setting
    const QBrush backgroundBrush = QBrush(QColor(0xff, 0xff, 0xff));
    const QPen   sliderPen       = QPen(QBrush(QColor(20, 20, 20)), 3.0);
    const QBrush sliderBrush    = QBrush(QColor(20, 20, 20));

    // Background
    painter.setRenderHint(QPainter::Antialiasing);
    painter.fillRect(rect, backgroundBrush);

#if 0
    QImage image(w, h, QImage::Format_ARGB32);

    for (int pix_x = 0; pix_x < w; ++ pix_x)
    {
        for (int pix_y = 0; pix_y < h; ++ pix_y)
        {
            const double x0 = static_cast<double>(pix_x) / static_cast<double>(w);
            const double x1 = static_cast<double>(pix_y) / static_cast<double>(h);
            VectorXd x(2); x << x0, x1;

            const double   y     = core.evaluateObjectiveFunction(x);
            const VectorXd color = ColorUtility::getHeatmapColor(y);

            image.setPixel(pix_x, pix_y, qRgba(color(0) * 255, color(1) * 255, color(2) * 255, 255));
        }
    }
    painter.drawImage(0, 0, image);
#endif

    // Draw slider information
    {
        const double a0 = core.slider->end_0(0) * w;
        const double a1 = core.slider->end_0(1) * h;
        const double b0 = core.slider->end_1(0) * w;
        const double b1 = core.slider->end_1(1) * h;
        painter.setPen(sliderPen);
        painter.drawLine(QPointF(a0, a1), QPointF(b0, b1));
    }
    {
        const VectorXd x = core.computeParametersFromSlider(core.mainWindow->obtainSliderPosition());
        painter.setBrush(sliderBrush);
        painter.drawEllipse(QPointF(x(0) * w, x(1) * h), 4.0, 4.0);
    }
}
