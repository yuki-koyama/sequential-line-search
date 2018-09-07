#ifndef WIDGETMU_H
#define WIDGETMU_H

#include <QWidget>

class WidgetMu : public QWidget
{
    Q_OBJECT
public:
    explicit WidgetMu(QWidget *parent = 0);

protected:
    void paintEvent(QPaintEvent *event) Q_DECL_OVERRIDE;

signals:

public slots:

};

#endif // WIDGETMU_H
