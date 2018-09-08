#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

class QSlider;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    std::vector<QSlider*> sliders;

    double obtainSliderPosition() const;
    void updateRawSliders();

private slots:

    void on_actionClear_all_data_triggered();

    void on_actionProceed_optimization_triggered();

    void on_horizontalSlider_valueChanged(int value);

    void on_pushButton_clicked();

    void on_actionPrint_current_best_triggered();

    void on_actionExport_photos_on_slider_triggered();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
