#include "widgetpreview.hpp"
#include "core.hpp"
#include "mainwindow.hpp"
#include <Eigen/Core>
#include <QPaintEvent>
#include <QPainter>
#include <sequential-line-search/sequential-line-search.hpp>
#include <tinycolormap.hpp>

using Eigen::Vector3d;
using Eigen::VectorXd;

namespace
{
    Core& core = Core::getInstance();
}

WidgetPreview::WidgetPreview(QWidget* parent) : QWidget(parent)
{
    this->setFixedSize(320, 320);
}

void WidgetPreview::paintEvent(QPaintEvent* event)
{
    QPainter     painter(this);
    const QRect& rect = event->rect();

    const VectorXd x = core.optimizer->CalcPointFromSliderPosition(core.mainWindow->obtainSliderPosition());
    const double   y = core.evaluateObjectiveFunction(x);
    const auto     c = tinycolormap::GetJetColor(y).ConvertToQColor();

    const QBrush backgroundBrush = QBrush(c);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.fillRect(rect, backgroundBrush);
}
