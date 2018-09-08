#ifndef PREVIEWWIDGET_H
#define PREVIEWWIDGET_H

#include <QGLWidget>
#include <QImage>

class PreviewWidget : public QGLWidget
{
    Q_OBJECT
public:
    explicit PreviewWidget(QWidget *parent = 0);

    void setCurrentImage(const QImage &image);

    QSize sizeHint() const;

    const QImage& getImage() const { return image; }

signals:

public slots:

    void initializeGL();
    void paintGL();

private:
    QImage image;

    GLuint shaderProgram;
    GLuint texture;
    GLint  texLocation;
    GLint  p1Location;
    GLint  p2Location;
};

#endif // PREVIEWWIDGET_H
