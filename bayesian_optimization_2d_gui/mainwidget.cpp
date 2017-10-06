#include "mainwidget.h"
#include <iostream>
#include <QPainter>
#include <QPaintEvent>
#include "core.h"
#include "gaussianprocessregressor.h"
#include "expectedimprovementmaximizer.h"
#include "colorutility.h"

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
    const QPen   mainLinePen     = QPen(QBrush(QColor(120, 0, 0)), 2.5);
    const QPen   EILinePen       = QPen(QBrush(QColor(20, 40, 100)), 1.5);
    const QPen   functionLinePen = QPen(QBrush(QColor(160, 160, 160)), 1.0);
    const QPen   dataPointPen    = QPen(QBrush(QColor(0, 0, 0)), 3.0);
    const QBrush dataPointBrush  = QBrush(QColor(0, 0, 0));
    const QPen   maximumPen      = QPen(QBrush(QColor(160, 20, 20)), 3.0);
    const QBrush maximumBrush    = QBrush(QColor(160, 20, 20));

    // Background
    painter.setRenderHint(QPainter::Antialiasing);
    painter.fillRect(rect, backgroundBrush);

    Eigen::MatrixXd val = Eigen::MatrixXd::Zero(w, h);
    for (int pix_x = 0; pix_x < w; ++ pix_x)
    {
        for (int pix_y = 0; pix_y < h; ++ pix_y)
        {
            if (core.regressor.get() == nullptr) continue;

            const double x0 = static_cast<double>(pix_x) / static_cast<double>(w);
            const double x1 = static_cast<double>(pix_y) / static_cast<double>(h);
            VectorXd x(2); x << x0, x1;
            val(pix_x, pix_y) = core.evaluateObjectiveFunction(x);
        }
    }
    if (std::abs(val.maxCoeff() - val.minCoeff()) > 1e-08)
    {
        val = (val - Eigen::MatrixXd::Constant(w, h, val.minCoeff())) / (val.maxCoeff() - val.minCoeff());
    }
    QImage image(w, h, QImage::Format_ARGB32);
    for (int pix_x = 0; pix_x < w; ++ pix_x)
    {
        for (int pix_y = 0; pix_y < h; ++ pix_y)
        {
            const VectorXd color = ColorUtility::getColor(val(pix_x, pix_y));
            image.setPixel(pix_x, pix_y, qRgba(color(0) * 255, color(1) * 255, color(2) * 255, 255));
        }
    }
    painter.drawImage(0, 0, image);

    // Variance and mean
//    std::vector<QPointF> variancePolygon;
//    std::vector<QPointF> mainPolyline;
//    for (int pix_x = 0; pix_x <= rect.width(); ++ pix_x)
//    {
//        const double x = static_cast<double>(pix_x) / static_cast<double>(rect.width());

//        const double y = core.regressor->estimate_y(VectorXd::Constant(1, x));
//        const double s = core.regressor->estimate_s(VectorXd::Constant(1, x));

//        const double pix_y = rect.height() - 0.5 * (rect.height() * (y + 1.0));
//        const double pix_s = 0.5 * (rect.height() * s);

//        variancePolygon.push_back(QPointF(pix_x, pix_y + pix_s));
//        mainPolyline.push_back(QPointF(pix_x, pix_y));
//    }
//    for (int pix_x = rect.width(); pix_x >= 0; -- pix_x)
//    {
//        const double x = static_cast<double>(pix_x) / static_cast<double>(rect.width());

//        const double y = core.regressor->estimate_y(VectorXd::Constant(1, x));
//        const double s = core.regressor->estimate_s(VectorXd::Constant(1, x));

//        const double pix_y = rect.height() - 0.5 * (rect.height() * (y + 1.0));
//        const double pix_s = 0.5 * (rect.height() * s);

//        variancePolygon.push_back(QPointF(pix_x, pix_y - pix_s));
//    }
//    painter.setBrush(QBrush(QColor(255, 200, 200), Qt::SolidPattern));
//    painter.setPen(QPen(Qt::NoPen));
//    painter.drawPolygon(&variancePolygon[0], variancePolygon.size());
//    painter.setPen(mainLinePen);
//    painter.drawPolyline(&mainPolyline[0], mainPolyline.size());

//    // Expected Improvement
//    std::vector<QPointF> EIPolyline;
//    ExpectedImprovementMaximizer maximizer(core.regressor);
//    for (int pix_x = 0; pix_x <= rect.width(); ++ pix_x)
//    {
//        const double x     = static_cast<double>(pix_x) / static_cast<double>(rect.width());
//        const double EI    = maximizer.calculateExpectedImprovedment(VectorXd::Constant(1, x));
//        const double pix_y = rect.height() - 2.0 * (rect.height() * EI);

//        EIPolyline.push_back(QPointF(pix_x, pix_y));
//    }
//    painter.setPen(EILinePen);
//    painter.drawPolyline(&EIPolyline[0], EIPolyline.size());

//    // Function
//    std::vector<QPointF> functionPolyline;
//    for (int pix_x = 0; pix_x <= rect.width(); ++ pix_x)
//    {
//        const double x     = static_cast<double>(pix_x) / static_cast<double>(rect.width());
//        const double y     = core.evaluateObjectiveFunction(VectorXd::Constant(1, x));
//        const double pix_y = rect.height() - 0.5 * (rect.height() * (y + 1.0));

//        functionPolyline.push_back(QPointF(pix_x, pix_y));
//    }
//    painter.setPen(functionLinePen);
//    painter.drawPolyline(&functionPolyline[0], functionPolyline.size());

//    // Data points
//    unsigned N = core.X.cols();
//    for (unsigned i = 0; i < N; ++ i)
//    {
//        const double x = core.X(0, i);
//        const double y = core.y(i);

//        const double pix_x = x * rect.width();
//        const double pix_y = rect.height() - 0.5 * (rect.height() * (y + 1.0));

//        painter.setPen(dataPointPen);
//        painter.setBrush(dataPointBrush);
//        painter.drawEllipse(QPointF(pix_x, pix_y), 4.0, 4.0);
//    }

//    // Maximum
//    if (!std::isnan(core.y_max))
//    {
//        const double pix_x = core.x_max(0) * rect.width();
//        const double pix_y = rect.height() - 0.5 * (rect.height() * (core.y_max + 1.0));

//        painter.setPen(maximumPen);
//        painter.setBrush(maximumBrush);
//        painter.drawEllipse(QPointF(pix_x, pix_y), 4.0, 4.0);
//    }
}
