#include "mainwindow.hpp"
#include "core.hpp"
#include "ui_mainwindow.h"
#include <QDir>
#include <QFileDialog>
#include <sequential-line-search/gaussian-process-regressor.hpp>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace
{
    Core& core = Core::getInstance();
} // namespace

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // Setup widgets
    ui->widget_y->content = MainWidget::Content::Objective;
    ui->widget_e->content = MainWidget::Content::ExpectedImprovement;
    ui->widget_m->content = MainWidget::Content::Mean;
    ui->widget_s->content = MainWidget::Content::StandardDeviation;
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_actionBatch_visualization_triggered()
{
    core.X     = MatrixXd::Constant(0, 0, 0.0);
    core.y     = VectorXd::Constant(0, 0.0);
    core.x_max = VectorXd::Constant(0, 0.0);
    core.y_max = NAN;
    core.computeRegression();
    ui->widget_y->update();

    constexpr unsigned n_iterations = 30;

    const QString path = QFileDialog::getExistingDirectory(this) + "/";

    for (unsigned i = 0; i < n_iterations; ++i)
    {
        core.proceedOptimization();
        window()->update();
        window()->grab().save(path + QString("window") + QString("%1").arg(core.y.rows(), 3, 10, QChar('0')) +
                              QString(".png"));
    }
}

void MainWindow::on_actionClear_all_data_triggered()
{
    core.X     = MatrixXd::Constant(0, 0, 0.0);
    core.y     = VectorXd::Constant(0, 0.0);
    core.x_max = VectorXd::Constant(0, 0.0);
    core.y_max = NAN;
    core.computeRegression();
    ui->widget_y->update();
    ui->widget_s->update();
    ui->widget_m->update();
    ui->widget_e->update();
}

void MainWindow::on_actionProceed_optimization_triggered()
{
    core.proceedOptimization();
    ui->widget_y->update();
    ui->widget_s->update();
    ui->widget_m->update();
    ui->widget_e->update();
}
