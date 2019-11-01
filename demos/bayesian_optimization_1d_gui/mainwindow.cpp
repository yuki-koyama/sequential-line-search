#include "mainwindow.hpp"
#include "core.hpp"
#include "ui_mainwindow.h"
#include <QDir>
#include <QFileDialog>
#include <iostream>
#include <sequential-line-search/gaussian-process-regressor.hpp>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace
{
    Core& core = Core::getInstance();
}

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->widget->setFixedHeight(600);
    ui->widget->setFixedWidth(1200);
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
    ui->widget->update();

    constexpr unsigned n_iterations = 15;

    const QString path = QFileDialog::getExistingDirectory(this) + "/";

    for (unsigned i = 0; i <= n_iterations; ++i)
    {
        ui->widget->grab().save(path + QString("%1").arg(core.y.rows(), 3, 10, QChar('0')) + QString(".png"));
        core.proceedOptimization();
        ui->widget->update();
    }
}

void MainWindow::on_actionClear_all_data_triggered()
{
    core.X     = MatrixXd::Constant(0, 0, 0.0);
    core.y     = VectorXd::Constant(0, 0.0);
    core.x_max = VectorXd::Constant(0, 0.0);
    core.y_max = NAN;
    core.computeRegression();
    ui->widget->update();
}

void MainWindow::on_actionProceed_optimization_triggered()
{
    core.proceedOptimization();
    ui->widget->update();
}
