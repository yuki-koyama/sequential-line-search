#ifndef MAINWIDGET_H
#define MAINWIDGET_H

#include <QWidget>

class MainWidget : public QWidget
{
    Q_OBJECT
public:
    explicit MainWidget(QWidget* parent = 0);

protected:
    void paintEvent(QPaintEvent* event) Q_DECL_OVERRIDE;

signals:

public slots:
};

#endif // MAINWIDGET_H
