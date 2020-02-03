#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <array>

namespace Ui
{
    class MainWindow;
}
class MainWidget;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = 0);
    ~MainWindow();

    double obtainSliderPosition() const;

private slots:

    void on_actionBatch_visualization_triggered();

    void on_actionClear_all_data_triggered();

    void on_actionProceed_optimization_triggered();

    void on_horizontalSlider_valueChanged(int value);

    void on_pushButton_clicked();

    void on_actionPrint_current_best_triggered();

private:
    Ui::MainWindow* ui;

    std::array<MainWidget*, 4> m_widgets;
};

#endif // MAINWINDOW_H
