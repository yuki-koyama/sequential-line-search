#include "colorutility.h"

using Eigen::Vector3d;
using Eigen::Matrix3d;
using namespace std;

// Most of magic numbers are taken from http://www.brucelindbloom.com/

namespace
{
    inline double rgb2h(const Vector3d& rgb)
    {
        const double r = rgb(0);
        const double g = rgb(1);
        const double b = rgb(2);
        const double M = std::max({r, g, b});
        const double m = std::min({r, g, b});
        
        double h;
        if (M == m)      h = 0.0;
        else if (m == b) h = 60.0 * (g - r) / (M - m) + 60.0;
        else if (m == r) h = 60.0 * (b - g) / (M - m) + 180.0;
        else if (m == g) h = 60.0 * (r - b) / (M - m) + 300.0;
        else             abort();
        h /= 360.0;
        if (h < 0.0) {
            ++ h;
        } else if (h > 1.0) {
            -- h;
        }
        return h;
    }
    
    inline double rgb2s4hsv(const Vector3d& rgb)
    {
        const double r = rgb(0);
        const double g = rgb(1);
        const double b = rgb(2);
        const double M = std::max({r, g, b});
        const double m = std::min({r, g, b});
        
        if (M < 1e-14) return 0.0;
        return (M - m) / M;
    }
    
    inline double rgb2s4hsl(const Vector3d& rgb)
    {
        const double r = rgb(0);
        const double g = rgb(1);
        const double b = rgb(2);
        const double M = std::max({r, g, b});
        const double m = std::min({r, g, b});
        
        if (M - m < 1e-14) return 0.0;
        return (M - m) / (1.0 - abs(M + m - 1.0));
    }
}

namespace ColorUtility
{
    // This implementation is based on https://gist.github.com/liovch/3168961
    Vector3d hsl2rgb(const Vector3d& hsl)
    {
        auto hue2rgb = [] (const double f1, const double f2, double hue)
        {
            if (hue < 0.0) hue += 1.0;
            if (hue > 1.0) hue -= 1.0;
            
            double res;
            if ((6.0 * hue) < 1.0)
                res = f1 + (f2 - f1) * 6.0 * hue;
            else if ((2.0 * hue) < 1.0)
                res = f2;
            else if ((3.0 * hue) < 2.0)
                res = f1 + (f2 - f1) * ((2.0 / 3.0) - hue) * 6.0;
            else
                res = f1;
            return res;
        };
        
        if (hsl.y() == 0.0)
        {
            return Vector3d(hsl.z(), hsl.z(), hsl.z());
        }
        
        const double f2 = (hsl.z() < 0.5) ? hsl.z() * (1.0 + hsl.y()) : (hsl.z() + hsl.y()) - (hsl.y() * hsl.z());
        const double f1 = 2.0 * hsl.z() - f2;
        
        Vector3d rgb;
        rgb(0) = hue2rgb(f1, f2, hsl.x() + (1.0 / 3.0));
        rgb(1) = hue2rgb(f1, f2, hsl.x());
        rgb(2) = hue2rgb(f1, f2, hsl.x() - (1.0 / 3.0));
        
        return rgb;
    }
    
    Vector3d rgb2hsv(const Vector3d &rgb)
    {
        const double r = rgb(0);
        const double g = rgb(1);
        const double b = rgb(2);
        const double M = std::max({r, g, b});
        
        const double h = rgb2h(rgb);
        const double s = rgb2s4hsv(rgb);
        const double v = M;
        
        return Vector3d(h, s, v);
    }
    
    double rgb2l(const Vector3d &rgb)
    {
        const double r = rgb(0);
        const double g = rgb(1);
        const double b = rgb(2);
        const double M = std::max({r, g, b});
        const double m = std::min({r, g, b});
        
        return 0.5 * (M + m);
    }
    
    Vector3d hsv2rgb(const Vector3d &hsv)
    {
        const double h = hsv(0);
        const double s = hsv(1);
        const double v = hsv(2);
        
        if (s < 1e-14)
        {
            return Vector3d(v, v, v);
        }
        
        const double h6 = h * 6.0;
        const int    i  = static_cast<int>(floor(h6)) % 6;
        const double f  = h6 - static_cast<double>(i);
        const double p  = v * (1 - s);
        const double q  = v * (1 - (s * f));
        const double t  = v * (1 - (s * (1 - f)));
        double r, g, b;
        switch(i)
        {
            case 0: r = v; g = t; b = p; break;
            case 1: r = q; g = v; b = p; break;
            case 2: r = p; g = v; b = t; break;
            case 3: r = p; g = q; b = v; break;
            case 4: r = t; g = p; b = v; break;
            case 5: r = v; g = p; b = q; break;
        }
        
        return Vector3d(r, g, b);
    }
    
    Vector3d rgb2hsl(const Vector3d &rgb)
    {
        const double h = rgb2h(rgb);
        const double s = rgb2s4hsl(rgb);
        const double l = rgb2l(rgb);
        
        return Vector3d(h, s, l);
    }
}
