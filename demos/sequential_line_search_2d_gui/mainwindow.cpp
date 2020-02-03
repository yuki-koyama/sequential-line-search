#include "mainwindow.hpp"
#include "core.hpp"
#include "mainwidget.hpp"
#include "ui_mainwindow.h"
#include <QDir>
#include <QFileDialog>
#include <QFutureWatcher>
#include <QLabel>
#include <QProgressDialog>
#include <QtConcurrent>
#include <fstream>
#include <iostream>
#include <nlopt-util.hpp>
#include <sequential-line-search/sequential-line-search.hpp>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using sequential_line_search::SequentialLineSearchOptimizer;

namespace
{
    Core& core = Core::getInstance();

    constexpr double a         = 0.500;
    constexpr double r         = 0.500;
    constexpr double b         = 0.001;
    constexpr double btl_scale = 0.010;
    constexpr double variance  = 0.100;

    constexpr int dimension = 2;

    constexpr bool use_slider_enlargement  = true;
    constexpr bool use_MAP_hyperparameters = true;
} // namespace

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent), ui(new Ui::MainWindow)
{
    core.mainWindow = this;
    ui->setupUi(this);

    core.optimizer =
        std::make_shared<SequentialLineSearchOptimizer>(dimension, use_slider_enlargement, use_MAP_hyperparameters);
    core.optimizer->SetHyperparams(a, r, b, variance, btl_scale);

    // Setup widgets
    for (int i : {0, 1, 2, 3})
    {
        constexpr unsigned s = 320;

        m_widgets[i] = new MainWidget(this);

        m_widgets[i]->setFixedSize(s, s);
    }

    m_widgets[0]->content           = MainWidget::Content::Objective;
    m_widgets[0]->draw_slider_space = true;
    m_widgets[0]->draw_slider_tick  = true;

    m_widgets[1]->content          = MainWidget::Content::ExpectedImprovement;
    m_widgets[1]->draw_maximum     = true;
    m_widgets[1]->draw_data_points = true;

    m_widgets[2]->content          = MainWidget::Content::Mean;
    m_widgets[2]->draw_maximum     = false;
    m_widgets[2]->draw_data_points = true;

    m_widgets[3]->content          = MainWidget::Content::StandardDeviation;
    m_widgets[3]->draw_maximum     = false;
    m_widgets[3]->draw_data_points = true;

    ui->gridLayout->addWidget(m_widgets[0], 1, 0);
    ui->gridLayout->addWidget(new QLabel("<center>Slider space</center>"), 2, 0);

    ui->gridLayout->addWidget(m_widgets[1], 1, 1);
    ui->gridLayout->addWidget(new QLabel("<center>Expected improvement</center>"), 2, 1);

    ui->gridLayout->addWidget(m_widgets[2], 3, 0);
    ui->gridLayout->addWidget(new QLabel("<center>Predicted mean</center>"), 4, 0);

    ui->gridLayout->addWidget(m_widgets[3], 3, 1);
    ui->gridLayout->addWidget(new QLabel("<center>Predicted stdev</center>"), 4, 1);

    ui->horizontalSlider->setValue((ui->horizontalSlider->maximum() + ui->horizontalSlider->minimum()) / 2);
}

MainWindow::~MainWindow()
{
    delete ui;
}

double MainWindow::obtainSliderPosition() const
{
    return static_cast<double>(ui->horizontalSlider->value() - ui->horizontalSlider->minimum()) /
           static_cast<double>(ui->horizontalSlider->maximum() - ui->horizontalSlider->minimum());
}

namespace
{

    double objectiveFunc(const std::vector<double>& x, std::vector<double>&, void*)
    {
        return core.evaluateObjectiveFunction(Eigen::Map<const Eigen::VectorXd>(&x[0], x.size()));
    }

} // namespace

void MainWindow::on_actionBatch_visualization_triggered()
{
    constexpr unsigned n_iterations = 10;

    const unsigned orig_min = ui->horizontalSlider->minimum();
    const unsigned orig_max = ui->horizontalSlider->maximum();

    ui->horizontalSlider->setMinimum(0);
    ui->horizontalSlider->setMaximum(10000);

    const QString path = QFileDialog::getExistingDirectory(this) + "/";

    std::ofstream ofs(path.toStdString() + "residuals.csv");

    auto background_process = [&]() {
        const Eigen::Vector2d x_opt = nloptutil::solve(Eigen::Vector2d(0.5, 0.5),
                                                       Eigen::Vector2d(1.0, 1.0),
                                                       Eigen::Vector2d(0.0, 0.0),
                                                       objectiveFunc,
                                                       nlopt::LN_COBYLA,
                                                       nullptr);

        MainWidget* widget_y = m_widgets[0];
        MainWidget* widget_e = m_widgets[1];
        MainWidget* widget_m = m_widgets[2];
        MainWidget* widget_s = m_widgets[3];

        widget_y->draw_slider_space = false;
        widget_y->draw_slider_tick  = false;
        widget_y->grab().save(path + QString("y.png"));
        widget_y->draw_slider_space = true;
        widget_y->draw_slider_tick  = true;

        for (unsigned i = 0; i <= n_iterations; ++i)
        {
            // search the best position
            int    max_slider = -1;
            double max_y      = -1e+10;
            for (int j = ui->horizontalSlider->minimum(); j < ui->horizontalSlider->maximum(); ++j)
            {
                ui->horizontalSlider->setValue(j);
                const double y =
                    core.evaluateObjectiveFunction(core.optimizer->CalcPointFromSliderPosition(obtainSliderPosition()));
                if (y > max_y)
                {
                    max_y      = y;
                    max_slider = j;
                }
            }

            ofs << i << "," << (core.optimizer->GetMaximizer() - x_opt).norm() << std::endl;

            ui->horizontalSlider->setValue(max_slider);
            window()->grab().save(path + QString("window") + QString("%1").arg(i, 3, 10, QChar('0')) + QString(".png"));
            widget_e->grab().save(path + QString("e") + QString("%1").arg(i, 3, 10, QChar('0')) + QString(".png"));
            widget_m->grab().save(path + QString("m") + QString("%1").arg(i, 3, 10, QChar('0')) + QString(".png"));
            widget_s->grab().save(path + QString("s") + QString("%1").arg(i, 3, 10, QChar('0')) + QString(".png"));
            widget_y->grab().save(path + QString("y") + QString("%1").arg(i, 3, 10, QChar('0')) + QString(".png"));

            widget_e->draw_maximum = false;
            widget_e->grab().save(path + QString("_e") + QString("%1").arg(i, 3, 10, QChar('0')) + QString(".png"));
            widget_e->draw_maximum = true;

            widget_y->draw_slider_tick = false;
            widget_y->grab().save(path + QString("_y") + QString("%1").arg(i, 3, 10, QChar('0')) + QString(".png"));
            widget_y->draw_slider_tick = true;

            core.optimizer->SubmitLineSearchResult(obtainSliderPosition());
        }
    };

    QProgressDialog      dialog(QString(), QString(), 0, 0, this);
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
    core.optimizer =
        std::make_shared<SequentialLineSearchOptimizer>(dimension, use_slider_enlargement, use_MAP_hyperparameters);
    core.optimizer->SetHyperparams(a, r, b, variance, btl_scale);

    for (auto widget : m_widgets)
    {
        widget->update();
    }
}

void MainWindow::on_actionProceed_optimization_triggered()
{
    core.optimizer->SubmitLineSearchResult(obtainSliderPosition());

    for (auto widget : m_widgets)
    {
        widget->update();
    }
}

void MainWindow::on_horizontalSlider_valueChanged(int /*value*/)
{
    m_widgets[0]->update();
    ui->widget_preview->update();
}

void MainWindow::on_pushButton_clicked()
{
    core.optimizer->SubmitLineSearchResult(obtainSliderPosition());

    ui->horizontalSlider->setValue((ui->horizontalSlider->maximum() + ui->horizontalSlider->minimum()) / 2);
    for (auto widget : m_widgets)
    {
        widget->update();
    }
}

void MainWindow::on_actionPrint_current_best_triggered()
{
    std::cout << core.optimizer->GetMaximizer().transpose() << std::endl;
}
