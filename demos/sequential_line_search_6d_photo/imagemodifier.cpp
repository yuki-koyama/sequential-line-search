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

namespace ImageModifier
{
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
            rgbArray = enhancer::internal::changeColorBalance(rgbArray, balance);
            
            // brightness
            for (int k = 0; k < 3; ++ k) rgbArray[k] *= 1.0 + brightness;
            
            // contrast
            for (int k = 0; k < 3; ++ k) rgbArray[k] = (rgbArray[k] - 0.5) * (tan((contrast + 1.0) * M_PI_4) ) + 0.5;
            
            // clamp
            for (int k = 0; k < 3; ++ k) rgbArray[k] = enhancer::internal::clamp(rgbArray[k]);
            
            // saturation
            Vector3d hsvVector = enhancer::internal::rgb2hsv(rgbArray);
            double s = hsvVector.y();
            s *= saturation + 1.0;
            hsvVector(1) = enhancer::internal::clamp(s);
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
}
