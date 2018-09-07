#define MULTI_THREAD

#include "imagemodifier.h"
#include <cmath>
#include <cassert>
#ifdef MULTI_THREAD
#include <thread>
#endif
#include <QImage>
#include <enhancer.hpp>

using std::vector;
using std::max;
using std::min;
using std::thread;
using Eigen::Vector3d;

namespace ImageModifier {

extern Vector3d changeColorBalance(const Vector3d& inputRgb, const Vector3d& shift);

inline float clamp(const float value) { return max(0.0, min(static_cast<double>(value), 1.0)); }
inline Vector3d clamp(const Vector3d& v) { return Vector3d(clamp(v.x()), clamp(v.y()), clamp(v.z())); }

inline Vector3d qRgb2rgb(const QRgb& qRgb)
{
    const int r = qRed(qRgb);
    const int g = qGreen(qRgb);
    const int b = qBlue(qRgb);
    Vector3d rgb(r, g, b);
    return rgb / 255.0;
}

inline std::vector<double> convert(const Eigen::VectorXd& x)
{
    std::vector<double> _x(x.rows());
    Eigen::Map<Eigen::VectorXd>(&_x[0], x.rows()) = x;
    return _x;
}

QImage modifyImage(const QImage &image, const Eigen::VectorXd &set)
{
    return modifyImage(image, convert(set));
}

QImage modifyImage(const QImage& image, const std::vector<double>& set)
{
    assert (set.size() == 3 || set.size() == 6);

    const double brightness = set[0] - 0.5;
    const double contrast   = set[1] - 0.5;
    const double saturation = set[2] - 0.5;
    Vector3d balance;
    for (int i = 0; i < 3; ++ i) {
        if (set.size() == 3) balance[i] = 0.5        - 0.5;
        if (set.size() == 6) balance[i] = set[i + 3] - 0.5;
    }

    const int w = image.rect().width();
    const int h = image.rect().height();

    QImage newImg = QImage(w, h, QImage::Format_RGB32);

    auto changePixelColor = [&] (const int i, const int j)
    {
        QRgb rgb = image.pixel(i, j);
        Vector3d rgbArray = qRgb2rgb(rgb);

        // color balance
        rgbArray = changeColorBalance(rgbArray, balance);

        // brightness
        for (int k = 0; k < 3; ++ k) rgbArray[k] *= 1.0 + brightness;

        // contrast
        for (int k = 0; k < 3; ++ k) rgbArray[k] = (rgbArray[k] - 0.5) * (tan((contrast + 1.0) * M_PI_4) ) + 0.5;

        // clamp
        for (int k = 0; k < 3; ++ k) rgbArray[k] = clamp(rgbArray[k]);

        // saturation
        Vector3d hsvVector = enhancer::internal::rgb2hsv(rgbArray);
        double s = hsvVector.y();
        s *= saturation + 1.0;
        hsvVector(1) = clamp(s);
        const Vector3d rgbVector = enhancer::internal::hsv2rgb(hsvVector);

        rgb = qRgb(static_cast<int>(rgbVector(0) * 255.0),
                   static_cast<int>(rgbVector(1) * 255.0),
                   static_cast<int>(rgbVector(2) * 255.0));

        newImg.setPixel(i, j, rgb);
    };

#ifdef MULTI_THREAD
    vector<thread> ts;
    for (int y = 0; y < h; ++ y)
    {
        ts.push_back(thread([changePixelColor, w] (const int y)
        {
            for (int x = 0; x < w; ++ x)
            {
                changePixelColor(x, y);
            }
        }, y));
    }
    for (thread& t : ts) t.join();
#else
    for (int x = 0; x < w; ++ x)
    {
         for (int y = 0; y < h; ++ y)
         {
             changePixelColor(x, y);
         }
    }
#endif

    return newImg;
}

// The algorithm is taken from https://gist.github.com/liovch/3168961
Vector3d changeColorBalance(const Vector3d& inputRgb, const Vector3d& shift)
{
    const double   a         = 0.250;
    const double   b         = 0.333;
    const double   scale     = 0.700;

    const double   lightness = enhancer::internal::rgb2l(inputRgb);
    const Vector3d midtones  = (clamp((lightness - b) / a + 0.5) * clamp((lightness + b - 1.0) / (- a) + 0.5) * scale) * shift;
    const Vector3d newColor  = clamp(inputRgb + midtones);
    const Vector3d newHsl    = enhancer::internal::rgb2hsl(newColor);

    return enhancer::internal::hsl2rgb(Vector3d(newHsl(0), newHsl(1), lightness));
}

}
