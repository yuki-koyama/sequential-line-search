#ifndef WIDGETPREVIEW_H
#define WIDGETPREVIEW_H

#include <QWidget>

class WidgetPreview : public QWidget
{
    Q_OBJECT
public:
    explicit WidgetPreview(QWidget *parent = 0);

protected:
    void paintEvent(QPaintEvent *event) Q_DECL_OVERRIDE;

signals:

public slots:

};

#endif // WIDGETPREVIEW_H
