#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>
#include <memory>
#include <QDir>
#include <QImage>
#include <sequential-line-search/sequential-line-search.h>
#include "core.h"
#include "previewwidget.h"
#include "imagemodifier.h"
#include "directoryutility.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace
{
    Core& core = Core::getInstance();
}

MainWindow::MainWindow(QWidget *parent) :
QMainWindow(parent),
ui(new Ui::MainWindow)
{
    core.mainWindow = this;
    ui->setupUi(this);
    
    // Set Widgets
    ui->widget_y->content = MainWidget::Content::None;
    ui->widget_y->draw_slider_space = true;
    ui->widget_y->draw_slider_tick  = true;
    
    ui->widget_e->content = MainWidget::Content::ExpectedImprovement;
    ui->widget_e->draw_maximum      = true;
    ui->widget_e->draw_data_points  = true;
    
    ui->widget_m->content = MainWidget::Content::Mean;
    ui->widget_m->draw_maximum      = true;
    ui->widget_m->draw_data_points  = true;
    
    ui->widget_s->content = MainWidget::Content::StandardDeviation;
    ui->widget_s->draw_maximum      = true;
    ui->widget_s->draw_data_points  = true;
    
    // Set a target photo
    const std::string photo_name = SEQUENTIAL_LINE_SEARCH_PHOTO_NAME;
    const QImage      image      = QImage((DirectoryUtility::getResourceDirectory() + "/data/" + photo_name).c_str());
    ui->widget_preview->setCurrentImage(image.scaledToWidth(std::min(1600, image.width())));
    
    // Generate sliders for visualization
    std::vector<std::string> names;
    for (unsigned i = 0; i < core.dim; ++ i) { names.push_back("x" + std::to_string(i)); }
    
    if (core.dim == 6)
    {
        names = std::vector<std::string>{
            "Brightness",
            "Contrast",
            "Saturation",
            "Color Balance [R]",
            "Color Balance [G]",
            "Color Balance [B]"
        };
    } else if (core.dim == 2)
    {
        names = std::vector<std::string>{
            "Saturation",
            "Color Balance [B]",
        };
    }
    assert(core.dim == names.size());
    
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
    
    if (core.dim != 2)
    {
        ui->widget_y->setVisible(false);
        ui->widget_e->setVisible(false);
        ui->widget_m->setVisible(false);
        ui->widget_s->setVisible(false);
        ui->label_y->setVisible(false);
        ui->label_e->setVisible(false);
        ui->label_m->setVisible(false);
        ui->label_s->setVisible(false);
    }
    
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
    // Proceed optimization step
    core.proceedOptimization();
    
    // Damp data
    core.regressor->dampData(DirectoryUtility::getTemporaryDirectory());
    
    // Reset slider position
    ui->horizontalSlider->setValue((ui->horizontalSlider->maximum() - ui->horizontalSlider->minimum()) / 2 + ui->horizontalSlider->minimum());
    
    // Repaint evary widget
    ui->widget_s->repaint();
    ui->widget_m->repaint();
    ui->widget_e->repaint();
    ui->horizontalSlider->repaint();
    ui->widget_preview->repaint();
}

void MainWindow::on_horizontalSlider_valueChanged(int /*value*/)
{
    ui->widget_y->update();
    ui->widget_preview->update();
    updateRawSliders();
}

void MainWindow::on_pushButton_clicked()
{
    on_actionProceed_optimization_triggered();
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
    constexpr unsigned n = 10;
    constexpr unsigned w = 640;
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
