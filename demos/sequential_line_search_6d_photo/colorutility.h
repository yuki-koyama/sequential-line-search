#ifndef COLORUTILITY_H
#define COLORUTILITY_H

#include <Eigen/Core>

// Note: All the RGB colors should be in [0, 1] and described in sRGB

// RGB ... sRGB in [0, 1]
// rgb ... normalized rgb (i.e., inverse companding)

namespace ColorUtility
{
    inline double getLuminance(Eigen::Vector3d rgb)
    {
        return 0.298912 * rgb(0) + 0.586611 * rgb(1) + 0.114478 * rgb(2);
    }
    
    Eigen::Vector3d rgb2XYZ(const Eigen::Vector3d& rgb);
    Eigen::Vector3d XYZ2rgb(const Eigen::Vector3d& XYZ);
    Eigen::Vector3d rgb2hsv(const Eigen::Vector3d& rgb);
    Eigen::Vector3d rgb2hsl(const Eigen::Vector3d& rgb);
    Eigen::Vector3d hsv2rgb(const Eigen::Vector3d& hsv);
    Eigen::Vector3d hsl2rgb(const Eigen::Vector3d& hsl);
    
    // L* is roughly in [0, 100], and a*b* are roughly in [-100, 100]
    Eigen::Vector3d XYZ2Lab(const Eigen::Vector3d& XYZ, const Eigen::Vector3d &referenceXYZ = Eigen::Vector3d(95.047, 100.000, 108.883));
    
    double rgb2l(const Eigen::Vector3d& rgb);
    
    inline Eigen::Vector3d rgb2Lab(const Eigen::Vector3d& rgb)
    {
        return XYZ2Lab(rgb2XYZ(rgb));
    }
}

#endif // COLORUTILITY_H
