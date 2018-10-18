#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>
#include <fstream>
#include <QDir>
#include <QFileDialog>
#include <QProgressDialog>
#include <QFutureWatcher>
#include <QtConcurrent>
#include <nlopt-util.hpp>
#include "core.h"
#include "preferenceregressor.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace{
Core& core = Core::getInstance();

constexpr double   a          = 0.500;
constexpr double   r          = 0.500;
constexpr double   noise      = 0.001;
constexpr double   btl_scale  = 0.010;
constexpr double   variance   = 0.100;

constexpr bool     use_MAP    = true;
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    core.mainWindow = this;
    ui->setupUi(this);

    core.use_MAP_hyperparameters = use_MAP;

    PreferenceRegressor::Params::getInstance().a         = a;
    PreferenceRegressor::Params::getInstance().r         = r;
    PreferenceRegressor::Params::getInstance().variance  = variance;
    PreferenceRegressor::Params::getInstance().b         = noise;
    PreferenceRegressor::Params::getInstance().btl_scale = btl_scale;

    // Setup widgets
    ui->widget_y->content = MainWidget::Content::Objective;
    ui->widget_e->content = MainWidget::Content::ExpectedImprovement;
    ui->widget_m->content = MainWidget::Content::Mean;
    ui->widget_s->content = MainWidget::Content::StandardDeviation;

    ui->widget_y->draw_slider_space = true;
    ui->widget_y->draw_slider_tick  = true;

    ui->widget_e->draw_data_points  = true;
    ui->widget_e->draw_data_maximum = true;
    ui->widget_e->draw_maximum      = true;

    ui->widget_m->draw_data_points  = true;
    ui->widget_m->draw_data_maximum = true;

    ui->widget_s->draw_data_points  = true;
    ui->widget_s->draw_data_maximum = true;

    const unsigned s = 320;
    ui->widget_y->setFixedSize(s, s);
    ui->widget_s->setFixedSize(s, s);
    ui->widget_m->setFixedSize(s, s);
    ui->widget_e->setFixedSize(s, s);

    core.computeRegression();
    core.updateSliderEnds();
    ui->horizontalSlider->setValue((ui->horizontalSlider->maximum() + ui->horizontalSlider->minimum()) / 2);
}

MainWindow::~MainWindow()
{
    delete ui;
}

double MainWindow::obtainSliderPosition() const
{
    return static_cast<double>(ui->horizontalSlider->value() - ui->horizontalSlider->minimum()) / static_cast<double>(ui->horizontalSlider->maximum() - ui->horizontalSlider->minimum());
}

namespace
{

double objectiveFunc(const std::vector<double> &x, std::vector<double>&, void*)
{
    return core.evaluateObjectiveFunction(Eigen::Map<const Eigen::VectorXd>(&x[0], x.size()));
}

}

void MainWindow::on_actionBatch_visualization_triggered()
{
    core.clear();
    core.updateSliderEnds();

    constexpr unsigned n_iterations = 10;

    const unsigned orig_min = ui->horizontalSlider->minimum();
    const unsigned orig_max = ui->horizontalSlider->maximum();

    ui->horizontalSlider->setMinimum(    0);
    ui->horizontalSlider->setMaximum(10000);

    const QString path = QFileDialog::getExistingDirectory(this) + "/";

    std::ofstream ofs(path.toStdString() + "residuals.csv");

    auto background_process = [&]()
    {
        const Eigen::Vector2d x_opt = nloptutil::solve(Eigen::Vector2d(0.5, 0.5),
                                                       Eigen::Vector2d(1.0, 1.0),
                                                       Eigen::Vector2d(0.0, 0.0),
                                                       objectiveFunc,
                                                       nlopt::LN_COBYLA,
                                                       nullptr);
        
        ui->widget_y->draw_slider_space = false;
        ui->widget_y->draw_slider_tick  = false;
        ui->widget_y->grab().save(path + QString("y.png"));
        ui->widget_y->draw_slider_space = true;
        ui->widget_y->draw_slider_tick  = true;

        for (unsigned i = 0; i <= n_iterations; ++ i)
        {
            // search the best position
            int    max_slider = - 1;
            double max_y      = - 1e+10;
            for (int j = ui->horizontalSlider->minimum(); j < ui->horizontalSlider->maximum(); ++ j)
            {
                ui->horizontalSlider->setValue(j);
                const double y = core.evaluateObjectiveFunction(core.computeParametersFromSlider(j, ui->horizontalSlider->minimum(), ui->horizontalSlider->maximum()));
                if (y > max_y)
                {
                    max_y      = y;
                    max_slider = j;
                }
            }

            ofs << i << "," << (core.slider->orig_0 - x_opt).norm() << std::endl;

            ui->horizontalSlider->setValue(max_slider);
            window()->grab().save(path + QString("window") + QString("%1").arg(i, 3, 10, QChar('0')) + QString(".png"));
            ui->widget_e->grab().save(path + QString("e") + QString("%1").arg(i, 3, 10, QChar('0')) + QString(".png"));
            ui->widget_m->grab().save(path + QString("m") + QString("%1").arg(i, 3, 10, QChar('0')) + QString(".png"));
            ui->widget_s->grab().save(path + QString("s") + QString("%1").arg(i, 3, 10, QChar('0')) + QString(".png"));
            ui->widget_y->grab().save(path + QString("y") + QString("%1").arg(i, 3, 10, QChar('0')) + QString(".png"));

            ui->widget_e->draw_maximum = false;
            ui->widget_e->grab().save(path + QString("_e") + QString("%1").arg(i, 3, 10, QChar('0')) + QString(".png"));
            ui->widget_e->draw_maximum = true;

            ui->widget_y->draw_slider_tick = false;
            ui->widget_y->grab().save(path + QString("_y") + QString("%1").arg(i, 3, 10, QChar('0')) + QString(".png"));
            ui->widget_y->draw_slider_tick = true;

            core.proceedOptimization();
        }
    };

    QProgressDialog dialog(QString(), QString(), 0, 0, this);
    QFutureWatcher<void> watcher;
    QObject::connect(&watcher, SIGNAL(finished()), &dialog, SLOT(reset()));
    watcher.setFuture(QtConcurrent::run(background_process));
    dialog.exec();
    watcher.waitForFinished();

    ui->horizontalSlider->setMinimum(orig_min);
    ui->horizontalSlider->setMaximum(orig_max);
}

void MainWindow::on_actionClear_all_data_triggered()
{
    core.X     = MatrixXd::Constant(0, 0, 0.0);
    core.D.clear();

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

void MainWindow::on_horizontalSlider_valueChanged(int /*value*/)
{
    ui->widget_y->update();
    ui->widget_preview->update();
}

void MainWindow::on_pushButton_clicked()
{
    core.proceedOptimization();
    ui->horizontalSlider->setValue((ui->horizontalSlider->maximum() + ui->horizontalSlider->minimum()) / 2);
    ui->widget_y->update();
    ui->widget_s->update();
    ui->widget_m->update();
    ui->widget_e->update();
}

void MainWindow::on_actionPrint_current_best_triggered()
{
    std::cout << core.regressor->find_arg_max().transpose() << std::endl;
}
