#include "widgetpreview.hpp"
#include "core.hpp"
#include "mainwindow.hpp"
#include <Eigen/Core>
#include <QPaintEvent>
#include <QPainter>
#include <nlopt-util.hpp>
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

    // Search for the maximum and minimum values
    const auto upper  = Eigen::Vector2d::Ones();
    const auto lower  = Eigen::Vector2d::Zero();
    const auto x_init = 0.5 * (upper + lower);
    const auto x_max  = nloptutil::unconstrained::derivative_free::bounded::solve(
        x_init,
        upper,
        lower,
        [&](const Eigen::VectorXd& x) { return core.evaluateObjectiveFunction(x); },
        nlopt::GN_DIRECT,
        true,
        1000);
    const auto x_min = nloptutil::unconstrained::derivative_free::bounded::solve(
        x_init,
        upper,
        lower,
        [&](const Eigen::VectorXd& x) { return core.evaluateObjectiveFunction(x); },
        nlopt::GN_DIRECT,
        false,
        1000);

    m_f_min = core.evaluateObjectiveFunction(x_min);
    m_f_max = core.evaluateObjectiveFunction(x_max);
}

void WidgetPreview::paintEvent(QPaintEvent* event)
{
    QPainter     painter(this);
    const QRect& rect = event->rect();

    const VectorXd x = core.optimizer->CalcPointFromSliderPosition(core.mainWindow->obtainSliderPosition());
    const double   f = core.evaluateObjectiveFunction(x);
    const auto     c = tinycolormap::GetJetColor((f - m_f_min) / (m_f_max - m_f_min)).ConvertToQColor();

    const QBrush backgroundBrush = QBrush(c);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.fillRect(rect, backgroundBrush);
}
