#ifndef REGRESSOR_H
#define REGRESSOR_H

#include <Eigen/Core>
#include <vector>

namespace sequential_line_search
{
    class Regressor
    {
    public:
        virtual ~Regressor() {}

        unsigned GetNumDims() const { return getX().rows(); }

        virtual double PredictMu(const Eigen::VectorXd& x) const    = 0;
        virtual double PredictSigma(const Eigen::VectorXd& x) const = 0;

        virtual Eigen::VectorXd PredictMuDerivative(const Eigen::VectorXd& x) const    = 0;
        virtual Eigen::VectorXd PredictSigmaDerivative(const Eigen::VectorXd& x) const = 0;

        virtual Eigen::VectorXd GetKernelHyperparams() const = 0;

        virtual const Eigen::MatrixXd& getX() const = 0;
        virtual const Eigen::VectorXd& gety() const = 0;
        virtual double                 getb() const = 0;

        Eigen::VectorXd PredictMaximumPointFromData() const;
    };

    // k
    Eigen::VectorXd
    CalcSmallK(const Eigen::VectorXd& x, const Eigen::MatrixXd& X, const Eigen::VectorXd& kernel_hyperparameters);

    // K_y = K_f + sigma^{2} I
    Eigen::MatrixXd
    CalcLargeKY(const Eigen::MatrixXd& X, const Eigen::VectorXd& kernel_hyperparameters, const double noise_level);

    // K_f
    Eigen::MatrixXd CalcLargeKF(const Eigen::MatrixXd& X, const Eigen::VectorXd& kernel_hyperparameters);

    // partial k / partial x
    Eigen::MatrixXd CalcSmallKSmallXDerivative(const Eigen::VectorXd& x,
                                               const Eigen::MatrixXd& X,
                                               const Eigen::VectorXd& kernel_hyperparameters);

    // partial K_y / partial theta
    std::vector<Eigen::MatrixXd> CalcLargeKYThetaDerivative(const Eigen::MatrixXd& X,
                                                            const Eigen::VectorXd& kernel_hyperparameters);

    // partial K_y / partial sigma^{2}
    Eigen::MatrixXd CalcLargeKYNoiseLevelDerivative(const Eigen::MatrixXd& X,
                                                    const Eigen::VectorXd& kernel_hyperparameters,
                                                    const double           noise_level);

} // namespace sequential_line_search

#endif // REGRESSOR_H
