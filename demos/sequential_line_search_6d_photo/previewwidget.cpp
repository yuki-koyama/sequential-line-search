#include "previewwidget.h"

#include <string>
#include <QPainter>
#include "drawutility.h"
#include "core.h"
#include "utility.h"
#include "directoryutility.h"

using namespace std;

namespace {
Core& core = Core::getInstance();
}

PreviewWidget::PreviewWidget(QWidget *parent) :
    QGLWidget(parent)
{
}

QSize PreviewWidget::sizeHint() const
{
    return QSize(1280, 720);
}

void PreviewWidget::setCurrentImage(const QImage &image)
{
    this->image = image;
}

void PreviewWidget::initializeGL()
{
    glEnable(GL_MULTISAMPLE);
    
    static const string mainShaderName("enhancer");
    
    const string bundlePath             = DirectoryUtility::getResourceDirectory() + "/../Resources/shaders";
    const string mainVertexShaderPath   = bundlePath + "/" + mainShaderName + ".vs";
    const string mainFragmentShaderPath = bundlePath + "/" + mainShaderName + ".fs";
    
    // set shader program
    const int success = DrawUtility::loadShader(mainVertexShaderPath, mainFragmentShaderPath, &shaderProgram);
    if (success < 0) exit(1);
    
    texLocation = glGetUniformLocation(shaderProgram, "texture");
    p1Location  = glGetUniformLocation(shaderProgram, "first");
    p2Location  = glGetUniformLocation(shaderProgram, "second");
}

void PreviewWidget::paintGL()
{
    // Compute the viewport settings
    const GLint wRate = this->image.width();
    const GLint hRate = this->image.height();
    const GLint w = width() * devicePixelRatio();
    const GLint h = height() * devicePixelRatio();
    if (w * hRate == h * wRate)
    {
        glViewport(0, 0, w, h);
    }
    else if (w * hRate > h * wRate)
    {
        const int w_corrected = h * wRate / hRate;
        glViewport((w - w_corrected) / 2, 0, w_corrected, h);
    }
    else if (w * hRate < h * wRate)
    {
        const int h_corrected = w * hRate / wRate;
        glViewport(0, (h - h_corrected) / 2, w, h_corrected);
    }

    // Draw background and image
#if 0
    glClearColor(150.0 / 255.0, 150.0 / 255.0, 150.0 / 255.0, 1.0);
#else
    glClearColor(0.0, 0.0, 0.0, 1.0);
#endif
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    glUseProgram(shaderProgram);

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glEnable(GL_TEXTURE_2D);
    texture = bindTexture(this->image, GL_TEXTURE_2D, GL_RGB);
    glUniform1ui(texLocation, texture);

    const Eigen::VectorXd x = core.computeParametersFromSlider();
    if (x.rows() == 2)
    {
        glUniform3f(p1Location, 0.50, 0.50, x(0));
        glUniform3f(p2Location, 0.50, 0.50, x(1));
    }
    else if (x.rows() == 6)
    {
        glUniform3f(p1Location, x(0), x(1), x(2));
        glUniform3f(p2Location, x(3), x(4), x(5));
    }

    glBegin(GL_TRIANGLE_FAN);
    glTexCoord2d(0.0, 0.0); glVertex2d(-1.0, -1.0);
    glTexCoord2d(1.0, 0.0); glVertex2d(+1.0, -1.0);
    glTexCoord2d(1.0, 1.0); glVertex2d(+1.0, +1.0);
    glTexCoord2d(0.0, 1.0); glVertex2d(-1.0, +1.0);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glUseProgram(0);
}
