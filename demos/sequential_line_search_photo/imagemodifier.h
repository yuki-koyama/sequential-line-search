#ifndef IMAGEGENERATOR_H
#define IMAGEGENERATOR_H

#include <Eigen/Core>
#include <QImage>
#include <memory>
#include <vector>

namespace ImageModifier
{
    QImage modifyImage(const QImage& image, const std::vector<double>& set);
    QImage modifyImage(const QImage& image, const Eigen::VectorXd& set);
} // namespace ImageModifier

#endif // IMAGEGENERATOR_H
