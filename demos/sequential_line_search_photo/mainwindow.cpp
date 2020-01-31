#include "mainwindow.hpp"
#include "core.hpp"
#include "directoryutility.hpp"
#include "imagemodifier.hpp"
#include "mainwidget.hpp"
#include "ui_mainwindow.h"
#include <QDir>
#include <QImage>
#include <QLabel>
#include <QTimer>
#include <enhancer/enhancerwidget.hpp>
#include <iostream>
#include <memory>
#include <sequential-line-search/sequential-line-search.hpp>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using sequential_line_search::SequentialLineSearchOptimizer;

namespace
{
    Core& core = Core::getInstance();

    constexpr bool use_slider_enlargement  = true;
    constexpr bool use_MAP_hyperparameters = true;
} // namespace

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent), ui(new Ui::MainWindow), m_widgets({nullptr, nullptr, nullptr, nullptr})
{
    core.mainWindow = this;
    ui->setupUi(this);

    // Instantiate and set the preview widget
    enhancer_widget = new enhancer::EnhancerWidget(this);
    enhancer_widget->setMinimumSize(480, 320);
    ui->verticalLayout->insertWidget(0, enhancer_widget);

    // Instantiate visualization widgets if the target space is two-dimensional
    if (core.dim == 2)
    {
        for (int i : {0, 1, 2, 3})
        {
            m_widgets[i] = new MainWidget(this);

            m_widgets[i]->setFixedSize(160, 160);
        }

        m_widgets[0]->content           = MainWidget::Content::None;
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

        ui->gridLayout->setRowStretch(5, 0);
        ui->gridLayout->setRowStretch(5, 1);
    }

    // Set a target photo
    const std::string photo_name = SEQUENTIAL_LINE_SEARCH_PHOTO_NAME;
    const QImage      image      = QImage((DirectoryUtility::getResourceDirectory() + "/data/" + photo_name).c_str());
    enhancer_widget->setImage(image.scaledToWidth(std::min(1600, image.width())));

    // Generate sliders for visualization
    std::vector<std::string> names;
    for (unsigned i = 0; i < core.dim; ++i)
    {
        names.push_back("x" + std::to_string(i));
    }

    if (core.dim == 6)
    {
        names = std::vector<std::string>{
            "Brightness",
            "Contrast",
            "Saturation",
            "Color Balance [R]",
            "Color Balance [G]",
            "Color Balance [B]",
        };
    }
    else if (core.dim == 2)
    {
        names = std::vector<std::string>{
            "Saturation",
            "Color Balance [B]",
        };
    }
    assert(core.dim == names.size());

    for (unsigned i = 0; i < core.dim; ++i)
    {
        sliders.push_back(new QSlider(Qt::Horizontal, this));
        sliders[i]->setMinimumWidth(100);
        ui->formLayout->addRow(new QLabel(QString(names[i].c_str()), this), sliders[i]);
    }

    core.optimizer =
        std::make_shared<SequentialLineSearchOptimizer>(core.dim, use_slider_enlargement, use_MAP_hyperparameters);

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
    const QSlider* slider = ui->horizontalSlider;
    return static_cast<double>(slider->value() - slider->minimum()) /
           static_cast<double>(slider->maximum() - slider->minimum());
}

void MainWindow::updateRawSliders()
{
    const VectorXd x = core.optimizer->CalcPointFromSliderPosition(obtainSliderPosition());
    for (unsigned i = 0; i < core.dim; ++i)
    {
        QSlider* slider = sliders[i];
        slider->setValue(x(i) * (slider->maximum() - slider->minimum()) + slider->minimum());
    }
}

void MainWindow::on_actionClear_all_data_triggered()
{
    core.optimizer =
        std::make_shared<SequentialLineSearchOptimizer>(core.dim, use_slider_enlargement, use_MAP_hyperparameters);

    for (auto widget : m_widgets)
    {
        if (widget != nullptr)
        {
            widget->update();
        }
    }
}

void MainWindow::on_actionProceed_optimization_triggered()
{
    // Proceed optimization step
    core.optimizer->SubmitLineSearchResult(obtainSliderPosition());

    // Damp data
    core.optimizer->DampData(DirectoryUtility::getTemporaryDirectory());

    // Reset slider position
    ui->horizontalSlider->setValue((ui->horizontalSlider->maximum() - ui->horizontalSlider->minimum()) / 2 +
                                   ui->horizontalSlider->minimum());

    // Repaint evary widget
    for (auto widget : m_widgets)
    {
        if (widget != nullptr)
        {
            widget->update();
        }
    }
    ui->horizontalSlider->repaint();
    enhancer_widget->repaint();
}

void MainWindow::on_horizontalSlider_valueChanged(int /*value*/)
{
    if (m_widgets[0] != nullptr)
    {
        m_widgets[0]->update();
    }
    updateRawSliders();

    // Update the parameters for preview
    const VectorXd      x = core.optimizer->CalcPointFromSliderPosition(obtainSliderPosition());
    std::vector<double> parameters(6, 0.5);
    if (core.dim == 2)
    {
        parameters[2] = x(0);
        parameters[5] = x(1);
    }
    else
    {
        std::memcpy(parameters.data(), x.data(), sizeof(double) * 6);
    }
    enhancer_widget->setParameters(parameters);
    enhancer_widget->update();
}

void MainWindow::on_pushButton_clicked()
{
    on_actionProceed_optimization_triggered();
}

void MainWindow::on_actionPrint_current_best_triggered()
{
    std::cout << core.optimizer->GetMaximizer().transpose() << std::endl;
}

void MainWindow::on_actionExport_photos_on_slider_triggered()
{
    const std::string dir = DirectoryUtility::getTemporaryDirectory();
    const unsigned    m   = ui->horizontalSlider->minimum();
    const unsigned    M   = ui->horizontalSlider->maximum();

    // Export photos based on the current slider space
    constexpr unsigned n = 10;
    constexpr unsigned w = 640;
    for (unsigned i = 1; i <= n; ++i)
    {
        const unsigned        val             = m + (i - 1) * (M - m) / (n - 1);
        const double          slider_position = static_cast<double>(val - m) / static_cast<double>(M - m);
        const Eigen::VectorXd x               = core.optimizer->CalcPointFromSliderPosition(slider_position);
        const QImage          enhanced        = ImageModifier::modifyImage(enhancer_widget->getImage(), x);
        enhanced.save(QString((dir + "/full_" + std::to_string(i) + ".png").c_str()));
        enhanced.scaledToWidth(w, Qt::SmoothTransformation)
            .save(QString((dir + "/" + std::to_string(i) + ".png").c_str()));
    }

    // Export the current best photo
    if (core.optimizer->GetRawDataPoints().rows() != 0)
    {
        const Eigen::VectorXd x_best   = core.optimizer->GetMaximizer();
        const QImage          enhanced = ImageModifier::modifyImage(enhancer_widget->getImage(), x_best);
        enhanced.save(QString((dir + "/best.png").c_str()));
    }
}
