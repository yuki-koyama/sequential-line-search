#ifndef MAINWIDGET_H
#define MAINWIDGET_H

#include <QWidget>

class MainWidget : public QWidget
{
    Q_OBJECT
public:
    explicit MainWidget(QWidget* parent = 0);

    bool draw_slider_space = false;
    bool draw_slider_tick  = false;
    bool draw_data_points  = false;
    bool draw_data_maximum = false;
    bool draw_maximum      = false;

    enum class Content : long
    {
        Objective,
        Mean,
        StandardDeviation,
        ExpectedImprovement,
        None,
    };

    Content content = Content::None;

protected:
    void paintEvent(QPaintEvent* event) Q_DECL_OVERRIDE;

signals:

public slots:
};

#endif // MAINWIDGET_H
