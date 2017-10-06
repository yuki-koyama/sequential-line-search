#include "widgetpreview.h"
#include <QPainter>
#include <QPaintEvent>
#include <Eigen/Core>
#include "core.h"
#include "mainwindow.h"
#include "colorutility.h"

using Eigen::VectorXd;
using Eigen::Vector3d;

namespace
{
Core& core = Core::getInstance();
}

WidgetPreview::WidgetPreview(QWidget *parent) :
    QWidget(parent)
{
}

void WidgetPreview::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    const QRect& rect = event->rect();

    const VectorXd x = core.computeParametersFromSlider(core.mainWindow->obtainSliderPosition());
    const double   y = core.evaluateObjectiveFunction(x);
    const Vector3d c = ColorUtility::getColor(y);

    const QBrush backgroundBrush = QBrush(QColor(c(0) * 255.0, c(1) * 255.0, c(2) * 255.0));
    painter.setRenderHint(QPainter::Antialiasing);
    painter.fillRect(rect, backgroundBrush);
}
