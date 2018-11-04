#include "mainwidget.h"
#include <iostream>
#include <QPainter>
#include <QPaintEvent>
#include <tinycolormap.hpp>
#include <sequential-line-search/sequential-line-search.h>
#include "core.h"
#include "mainwindow.h"

using namespace sequential_line_search;
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
    const double slider_base_width = 8.0;
    const double slider_back_width = 2.0 * slider_base_width;
    const double knob_radius       = 8.0;
    const double knob_back_radius  = 1.2 * knob_radius;

    const QBrush backgroundBrush = QBrush(QColor(0xff, 0xff, 0xff));
    const QPen   sliderPen       = QPen(QBrush(QColor(20, 20, 20)), slider_base_width, Qt::SolidLine, Qt::RoundCap);
    const QPen   slider_back_pen = QPen(QBrush(QColor(220, 220, 220)), slider_back_width, Qt::SolidLine, Qt::RoundCap);
    const QBrush sliderBrush     = QBrush(QColor(20, 20, 20));
    const QBrush dataPointBrush  = QBrush(QColor(20, 20, 20));
    const QBrush maximumBrush    = QBrush(QColor(160, 20, 20));
    const double dot_radius      = 8.0;
    const double dot_back_radius = 1.4 * dot_radius;
    const QBrush dot_back_brush  = QBrush(QColor(220, 220, 220));
    const double dot_max_scale   = 1.4;

    // Background
    painter.setRenderHint(QPainter::Antialiasing);
    painter.fillRect(rect, backgroundBrush);

    Eigen::MatrixXd val = Eigen::MatrixXd::Zero(w, h);
    for (int pix_x = 0; pix_x < w; ++ pix_x)
    {
        for (int pix_y = 0; pix_y < h; ++ pix_y)
        {
            const double x0 = static_cast<double>(pix_x) / static_cast<double>(w);
            const double x1 = static_cast<double>(pix_y) / static_cast<double>(h);
            const Eigen::Vector2d x(x0, x1);

            switch (content) {
            case Content::Objective:
                val(pix_x, pix_y) = core.evaluateObjectiveFunction(x);
                break;
            case Content::Mean:
                val(pix_x, pix_y) = (core.regressor.get() != nullptr) ? core.regressor->estimate_y(x) : 0.0;
                break;
            case Content::StandardDeviation:
                val(pix_x, pix_y) = (core.regressor.get() != nullptr) ? core.regressor->estimate_s(x) : 0.0;
                break;
            case Content::ExpectedImprovement:
                val(pix_x, pix_y) = (core.regressor.get() != nullptr) ? acquisition_function::CalculateAcqusitionValue(*core.regressor, x) : 0.0;
                break;
            default:
                val(pix_x, pix_y) = 0.0;
                break;
            }
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
            const auto color = tinycolormap::GetJetColor(val(pix_x, pix_y));
            image.setPixel(pix_x, pix_y, qRgba(color[0] * 255, color[1] * 255, color[2] * 255, 255));
        }
    }
    painter.drawImage(0, 0, image);

    // Draw slider space
    if (draw_slider_space)
    {
        const double a0 = core.slider->end_0(0) * w;
        const double a1 = core.slider->end_0(1) * h;
        const double b0 = core.slider->end_1(0) * w;
        const double b1 = core.slider->end_1(1) * h;
        painter.setPen(slider_back_pen);
        painter.drawLine(QPointF(a0, a1), QPointF(b0, b1));
        painter.setPen(sliderPen);
        painter.drawLine(QPointF(a0, a1), QPointF(b0, b1));
    }

    // Draw slider tick position
    if (draw_slider_tick)
    {
        const VectorXd x = core.computeParametersFromSlider(core.mainWindow->obtainSliderPosition());
        painter.setPen(slider_back_pen);
        painter.drawEllipse(QPointF(x(0) * w, x(1) * h), knob_back_radius, knob_back_radius);
        painter.setPen(sliderPen);
        painter.setBrush(sliderBrush);
        painter.drawEllipse(QPointF(x(0) * w, x(1) * h), knob_radius, knob_radius);
    }

    // Data points
    if (draw_data_points)
    {
        unsigned N = core.data.X.cols();
        for (unsigned i = 0; i < N; ++ i)
        {
            const VectorXd x = core.data.X.col(i);
            const double pix_x = x(0) * rect.width();
            const double pix_y = x(1) * rect.height();
            painter.setPen(Qt::NoPen);
            painter.setBrush(dot_back_brush);
            painter.drawEllipse(QPointF(pix_x, pix_y), dot_back_radius, dot_back_radius);
        }
        for (unsigned i = 0; i < N; ++ i)
        {
            const VectorXd x = core.data.X.col(i);
            const double pix_x = x(0) * rect.width();
            const double pix_y = x(1) * rect.height();
            painter.setPen(Qt::NoPen);
            painter.setBrush(dataPointBrush);
            painter.drawEllipse(QPointF(pix_x, pix_y), dot_radius, dot_radius);
        }
    }

    // Maximum in data points
    if (draw_data_maximum && !std::isnan(core.y_max))
    {
        const double   pix_x = core.x_max(0) * rect.width();
        const double   pix_y = core.x_max(1) * rect.height();
        painter.setBrush(dot_back_brush);
        painter.drawEllipse(QPointF(pix_x, pix_y), dot_back_radius * dot_max_scale, dot_back_radius * dot_max_scale);
        painter.setBrush(maximumBrush);
        painter.drawEllipse(QPointF(pix_x, pix_y), dot_radius * dot_max_scale, dot_radius * dot_max_scale);
    }

    // Maximum in area
    if (draw_maximum && core.data.X.cols() != 0)
    {
        int pix_x;
        int pix_y;
        val.maxCoeff(&pix_x, &pix_y);

        painter.setBrush(dot_back_brush);
        painter.drawEllipse(QPointF(pix_x, pix_y), dot_back_radius * dot_max_scale, dot_back_radius * dot_max_scale);
        painter.setBrush(QBrush(QColor(20, 20, 160)));
        painter.drawEllipse(QPointF(pix_x, pix_y), dot_radius * dot_max_scale, dot_radius * dot_max_scale);
    }
}
