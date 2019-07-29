#include "widgetpreview.h"
#include "core.h"
#include "mainwindow.h"
#include <Eigen/Core>
#include <QPaintEvent>
#include <QPainter>
#include <sequential-line-search/sequential-line-search.h>
#include <tinycolormap.hpp>

using Eigen::Vector3d;
using Eigen::VectorXd;

namespace
{
    Core& core = Core::getInstance();
}

WidgetPreview::WidgetPreview(QWidget* parent) : QWidget(parent) {}

void WidgetPreview::paintEvent(QPaintEvent* event)
{
    QPainter     painter(this);
    const QRect& rect = event->rect();

    const VectorXd x = core.optimizer->GetParameters(core.mainWindow->obtainSliderPosition());
    const double   y = core.evaluateObjectiveFunction(x);
    const auto     c = tinycolormap::GetJetColor(y);

    const QBrush backgroundBrush = QBrush(QColor(c[0] * 255.0, c[1] * 255.0, c[2] * 255.0));
    painter.setRenderHint(QPainter::Antialiasing);
    painter.fillRect(rect, backgroundBrush);
}
