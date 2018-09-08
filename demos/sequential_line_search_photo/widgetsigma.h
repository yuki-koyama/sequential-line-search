#ifndef WIDGETSIGMA_H
#define WIDGETSIGMA_H

#include <QWidget>

class WidgetSigma : public QWidget
{
    Q_OBJECT
public:
    explicit WidgetSigma(QWidget *parent = 0);

protected:
    void paintEvent(QPaintEvent *event) Q_DECL_OVERRIDE;

signals:

public slots:

};

#endif // WIDGETSIGMA_H
