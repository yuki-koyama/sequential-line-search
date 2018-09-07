#ifndef COLORUTILITY_H
#define COLORUTILITY_H

#include <Eigen/Core>

// Note: All the RGB colors should be in [0, 1] and described in sRGB

// RGB ... sRGB in [0, 1]
// rgb ... normalized rgb (i.e., inverse companding)

namespace ColorUtility
{
    Eigen::Vector3d rgb2hsv(const Eigen::Vector3d& rgb);
    Eigen::Vector3d rgb2hsl(const Eigen::Vector3d& rgb);
    Eigen::Vector3d hsv2rgb(const Eigen::Vector3d& hsv);
    Eigen::Vector3d hsl2rgb(const Eigen::Vector3d& hsl);
    
    double rgb2l(const Eigen::Vector3d& rgb);
}

#endif // COLORUTILITY_H
