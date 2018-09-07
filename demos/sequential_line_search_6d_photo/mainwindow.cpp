#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>
#include <memory>
#include <QDir>
#include <QImage>
#include "core.h"
#include "preferenceregressor.h"
#include "previewwidget.h"
#include "utility.h"
#include "imagemodifier.h"
#include "directoryutility.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace{
Core& core = Core::getInstance();
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    core.mainWindow = this;
    ui->setupUi(this);

    // Set a target photo
    ui->widget_preview->setCurrentImage(QImage((DirectoryUtility::getResourceDirectory() + "data/1.jpg").c_str()));

    // Generate sliders for visualization
    std::vector<std::string> names;
    for (unsigned i = 0; i < core.dim; ++ i) { names.push_back("x" + std::to_string(i)); }

    if (core.dim == 6)
    {
        names = std::vector<std::string>{ "Brightness", "Contrast", "Saturation", "Color Balance [R]", "Color Balance [G]", "Color Balance [B]" };
    } else if (core.dim == 2)
    {
        names = std::vector<std::string>{ "Saturation", "Color Balance [B]" };
    }

    for (unsigned i = 0; i < core.dim; ++ i)
    {
        sliders.push_back(new QSlider(Qt::Horizontal, this));
        sliders[i]->setMinimumWidth(100);
        ui->formLayout->addRow(new QLabel(QString(names[i].c_str()), this), sliders[i]);
    }

    core.computeRegression();
    core.updateSliderEnds();
    ui->horizontalSlider->setValue((ui->horizontalSlider->maximum() + ui->horizontalSlider->minimum()) / 2);
    updateRawSliders();

    this->adjustSize();
}

MainWindow::~MainWindow()
{
    delete ui;
}

double MainWindow::obtainSliderPosition() const
{
    return static_cast<double>(ui->horizontalSlider->value() - ui->horizontalSlider->minimum()) / static_cast<double>(ui->horizontalSlider->maximum() - ui->horizontalSlider->minimum());
}

void MainWindow::updateRawSliders()
{
    const VectorXd x = core.computeParametersFromSlider();
    for (unsigned i = 0; i < core.dim; ++ i)
    {
        QSlider* slider = sliders[i];
        slider->setValue(x(i) * (slider->maximum() - slider->minimum()) + slider->minimum());
    }
}

void MainWindow::on_actionClear_all_data_triggered()
{
    core.X     = MatrixXd::Constant(0, 0, 0.0);
    core.D.clear();

    core.x_max = VectorXd::Constant(0, 0.0);
    core.y_max = NAN;

    core.updateSliderEnds();

    core.computeRegression();
    ui->widget_s->update();
    ui->widget_m->update();
    ui->widget_e->update();
}

void MainWindow::on_actionProceed_optimization_triggered()
{
    core.proceedOptimization();
    ui->widget_s->update();
    ui->widget_m->update();
    ui->widget_e->update();
}

void MainWindow::on_horizontalSlider_valueChanged(int /*value*/)
{
    ui->widget_y->update();
    ui->widget_preview->update();
    updateRawSliders();
}

void MainWindow::on_pushButton_clicked()
{
    core.proceedOptimization();
    ui->horizontalSlider->setValue((ui->horizontalSlider->maximum() + ui->horizontalSlider->minimum()) / 2);
    ui->widget_s->update();
    ui->widget_m->update();
    ui->widget_e->update();

    core.regressor->dampData(DirectoryUtility::getTemporaryDirectory());
}

void MainWindow::on_actionPrint_current_best_triggered()
{
    std::cout << core.regressor->find_arg_max().transpose() << std::endl;
}

void MainWindow::on_actionExport_photos_on_slider_triggered()
{
    const std::string dir = DirectoryUtility::getTemporaryDirectory();
    const unsigned m = ui->horizontalSlider->minimum();
    const unsigned M = ui->horizontalSlider->maximum();

    // Export photos based on the current slider space
    const unsigned n = 10;
    const unsigned w = 420;
    for (unsigned i = 1; i <= n; ++ i)
    {
        const unsigned val = m + (i - 1) * (M - m) / (n - 1);
        const Eigen::VectorXd x = core.computeParametersFromSlider(val, m, M);
        const QImage enhanced = ImageModifier::modifyImage(ui->widget_preview->getImage(), x);
        enhanced.save(QString((dir + "/full_" + std::to_string(i) + ".png").c_str()));
        enhanced.scaledToWidth(w, Qt::SmoothTransformation).save(QString((dir + "/" + std::to_string(i) + ".png").c_str()));
    }

    // Export the current best photo
    if (core.X.rows() != 0)
    {
        const Eigen::VectorXd x_best = core.regressor->find_arg_max();
        const QImage enhanced = ImageModifier::modifyImage(ui->widget_preview->getImage(), x_best);
        enhanced.save(QString((dir + "/best.png").c_str()));
    }
}
