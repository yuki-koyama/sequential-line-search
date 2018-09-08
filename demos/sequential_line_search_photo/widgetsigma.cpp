#include "widgetsigma.h"
#include <iostream>
#include <QPainter>
#include <QPaintEvent>
#include <tinycolormap.hpp>
#include "core.h"
#include "preferenceregressor.h"
#include "expectedimprovementmaximizer.h"

using Eigen::VectorXd;

namespace
{
    Core& core = Core::getInstance();
}

WidgetSigma::WidgetSigma(QWidget *parent) :
QWidget(parent)
{
    setAutoFillBackground(true);
}

void WidgetSigma::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    const QRect& rect = event->rect();
    const int    w    = rect.width();
    const int    h    = rect.height();
    
    // Draw setting
    const QBrush backgroundBrush = QBrush(QColor(0xff, 0xff, 0xff));
    const QPen   dataPointPen    = QPen(QBrush(QColor(0, 0, 0)), 3.0);
    const QBrush dataPointBrush  = QBrush(QColor(0, 0, 0));
    const QPen   maximumPen      = QPen(QBrush(QColor(0, 0, 0)), 3.0);
    const QBrush maximumBrush    = QBrush(QColor(160, 20, 20));
    
    // Background
    painter.setRenderHint(QPainter::Antialiasing);
    painter.fillRect(rect, backgroundBrush);
    
    QImage image(w, h, QImage::Format_ARGB32);
    for (int pix_x = 0; pix_x < w; ++ pix_x)
    {
        for (int pix_y = 0; pix_y < h; ++ pix_y)
        {
            const double x0 = static_cast<double>(pix_x) / static_cast<double>(w);
            const double x1 = static_cast<double>(pix_y) / static_cast<double>(h);
            VectorXd x = VectorXd::Constant(core.dim, 0.50); x(0) = x0; x(1) = x1;
            
            const double y     = core.regressor->estimate_s(x);
            const auto   color = tinycolormap::GetColor(y);
            
            image.setPixel(pix_x, pix_y, qRgba(color(0) * 255, color(1) * 255, color(2) * 255, 255));
        }
    }
    painter.drawImage(0, 0, image);
    
    // Data points
    unsigned N = core.X.cols();
    for (unsigned i = 0; i < N; ++ i)
    {
        const VectorXd x = core.X.col(i);
        
        const double pix_x = x(0) * rect.width();
        const double pix_y = x(1) * rect.height();
        
        painter.setPen(dataPointPen);
        painter.setBrush(dataPointBrush);
        painter.drawEllipse(QPointF(pix_x, pix_y), 4.0, 4.0);
    }
    
    // Maximum
    if (false && !std::isnan(core.y_max))
    {
        const double pix_x = core.x_max(0) * rect.width();
        const double pix_y = core.x_max(1) * rect.height();
        
        painter.setPen(maximumPen);
        painter.setBrush(maximumBrush);
        painter.drawEllipse(QPointF(pix_x, pix_y), 6.0, 6.0);
    }
}
