#ifndef WIDGETEI_H
#define WIDGETEI_H

#include <QWidget>

class WidgetEI : public QWidget
{
    Q_OBJECT
public:
    explicit WidgetEI(QWidget *parent = 0);

protected:
    void paintEvent(QPaintEvent *event) Q_DECL_OVERRIDE;

signals:

public slots:

};

#endif // WIDGETEI_H
