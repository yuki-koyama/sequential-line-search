#include "mainwidget.hpp"
#include "core.hpp"
#include <QPaintEvent>
#include <QPainter>
#include <iostream>
#include <sequential-line-search/acquisition-function.hpp>
#include <sequential-line-search/gaussian-process-regressor.hpp>

using namespace sequential_line_search;
using Eigen::VectorXd;

namespace
{
    Core& core = Core::getInstance();

    // Default : [- 1.0, + 1.0]
    const double offset_y = -1.3;
    const double scale_y  = +0.4;

    inline double val2pix_y(const double val_y, const int height)
    {
        return scale_y * (height - 0.5 * height * (val_y + offset_y));
    }

    inline double sd2pix_h(const double val_s, const int height)
    {
        // 1.96 * SD means 95% confidence interval
        return scale_y * 1.96 * 0.5 * height * val_s;
    }
} // namespace

MainWidget::MainWidget(QWidget* parent) : QWidget(parent)
{
    setAutoFillBackground(true);
}

void MainWidget::paintEvent(QPaintEvent* event)
{
    QPainter     painter(this);
    const QRect& rect = event->rect();

    // Draw setting
    const QBrush backgroundBrush = QBrush(QColor(0xf0, 0xf0, 0xf0));
    const QPen   mainLinePen     = QPen(QBrush(QColor(120, 0, 0)), 6.0);
    const QPen   EILinePen       = QPen(QBrush(QColor(20, 40, 100)), 6.0);
    const QPen   functionLinePen = QPen(QBrush(QColor(150, 150, 150)), 4.5, Qt::DashLine);
    const QPen   dataPointPen    = QPen(QBrush(QColor(0, 0, 0)), 6.0);
    const QBrush dataPointBrush  = QBrush(QColor(0, 0, 0));
    const QPen   maximumPen      = QPen(QBrush(QColor(160, 20, 20)), 6.0);
    const QBrush maximumBrush    = QBrush(QColor(160, 20, 20));

    // Background
    painter.setRenderHint(QPainter::Antialiasing);
    painter.fillRect(rect, backgroundBrush);

    if (!std::isnan(core.y_max))
    {
        // Variance and mean
        std::vector<QPointF> variancePolygon;
        std::vector<QPointF> mainPolyline;
        for (int pix_x = 0; pix_x <= rect.width(); ++pix_x)
        {
            const double x = static_cast<double>(pix_x) / static_cast<double>(rect.width());

            const double y = core.regressor->PredictMu(VectorXd::Constant(1, x));
            const double s = core.regressor->PredictSigma(VectorXd::Constant(1, x));

            const double pix_y = val2pix_y(y, rect.height());
            const double pix_s = sd2pix_h(s, rect.height());

            variancePolygon.push_back(QPointF(pix_x, pix_y + pix_s));
            mainPolyline.push_back(QPointF(pix_x, pix_y));
        }
        for (int pix_x = rect.width(); pix_x >= 0; --pix_x)
        {
            const double x = static_cast<double>(pix_x) / static_cast<double>(rect.width());

            const double y = core.regressor->PredictMu(VectorXd::Constant(1, x));
            const double s = core.regressor->PredictSigma(VectorXd::Constant(1, x));

            const double pix_y = val2pix_y(y, rect.height());
            const double pix_s = sd2pix_h(s, rect.height());

            variancePolygon.push_back(QPointF(pix_x, pix_y - pix_s));
        }
        painter.setBrush(QBrush(QColor(240, 200, 200), Qt::SolidPattern));
        painter.setPen(QPen(Qt::NoPen));
        painter.drawPolygon(&variancePolygon[0], variancePolygon.size());
        painter.setPen(mainLinePen);
        painter.drawPolyline(&mainPolyline[0], mainPolyline.size());

        // Expected Improvement
        std::vector<QPointF> EIPolyline;
        std::vector<QPointF> EIPolygon;
        VectorXd             EIs(rect.width() + 1);
        for (int pix_x = 0; pix_x <= rect.width(); ++pix_x)
        {
            const double x  = static_cast<double>(pix_x) / static_cast<double>(rect.width());
            const double EI = acquisition_func::CalcAcqusitionValue(
                *core.regressor, VectorXd::Constant(1, x), AcquisitionFuncType::ExpectedImprovement);
            EIs(pix_x) = EI;
        }
        EIs /= EIs.maxCoeff();
        EIs *= 0.12;
        for (int pix_x = 0; pix_x <= rect.width(); ++pix_x)
        {
            const double offset = -2.0;
            const double pix_y  = offset + rect.height() - 2.0 * (rect.height() * EIs(pix_x));
            EIPolyline.push_back(QPointF(pix_x, pix_y));
            EIPolygon.push_back(QPointF(pix_x, pix_y));
        }
        EIPolygon.push_back(QPointF(rect.width(), rect.height()));
        EIPolygon.push_back(QPointF(0.0, rect.height()));
        painter.setPen(EILinePen);
        painter.drawPolyline(&EIPolyline[0], EIPolyline.size());
        painter.setPen(QPen(Qt::NoPen));
        painter.setBrush(QBrush(QColor(190, 210, 240), Qt::SolidPattern));
        painter.drawPolygon(&EIPolygon[0], EIPolygon.size());
    }

    // Function
    std::vector<QPointF> functionPolyline;
    for (int pix_x = 0; pix_x <= rect.width(); ++pix_x)
    {
        const double x     = static_cast<double>(pix_x) / static_cast<double>(rect.width());
        const double y     = core.evaluateObjectiveFunction(VectorXd::Constant(1, x));
        const double pix_y = val2pix_y(y, rect.height());

        functionPolyline.push_back(QPointF(pix_x, pix_y));
    }
    painter.setPen(functionLinePen);
    painter.drawPolyline(&functionPolyline[0], functionPolyline.size());

    // Data points
    unsigned N = core.X.cols();
    for (unsigned i = 0; i < N; ++i)
    {
        const double x = core.X(0, i);
        const double y = core.y(i);

        const double pix_x = x * rect.width();
        const double pix_y = val2pix_y(y, rect.height());

        painter.setPen(dataPointPen);
        painter.setBrush(dataPointBrush);
        painter.drawEllipse(QPointF(pix_x, pix_y), 8.0, 8.0);
    }

    // Maximum
    if (!std::isnan(core.y_max))
    {
        const double pix_x = core.x_max(0) * rect.width();
        const double pix_y = val2pix_y(core.y_max, rect.height());

        painter.setPen(maximumPen);
        painter.setBrush(maximumBrush);
        painter.drawEllipse(QPointF(pix_x, pix_y), 8.0, 8.0);
    }
}
