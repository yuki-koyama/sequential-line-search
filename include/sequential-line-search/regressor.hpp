#ifndef SEQUENTIAL_LINE_SEARCH_REGRESSOR_HPP
#define SEQUENTIAL_LINE_SEARCH_REGRESSOR_HPP

#include <Eigen/Core>
#include <vector>

namespace sequential_line_search
{
    enum class KernelType
    {
        ArdSquaredExponentialKernel,
        ArdMatern52Kernel,
    };

    using Kernel                   = double (*)(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&);
    using KernelThetaDerivative    = Eigen::VectorXd (*)(const Eigen::VectorXd&,
                                                      const Eigen::VectorXd&,
                                                      const Eigen::VectorXd&);
    using KernelFirstArgDerivative = Eigen::VectorXd (*)(const Eigen::VectorXd&,
                                                         const Eigen::VectorXd&,
                                                         const Eigen::VectorXd&);

    class Regressor
    {
    public:
        Regressor(const KernelType kernel_type);
        virtual ~Regressor() {}

        unsigned GetNumDims() const { return GetLargeX().rows(); }

        virtual double PredictMu(const Eigen::VectorXd& x) const    = 0;
        virtual double PredictSigma(const Eigen::VectorXd& x) const = 0;

        virtual Eigen::VectorXd PredictMuDerivative(const Eigen::VectorXd& x) const    = 0;
        virtual Eigen::VectorXd PredictSigmaDerivative(const Eigen::VectorXd& x) const = 0;

        virtual const Eigen::VectorXd& GetKernelHyperparams() const = 0;
        virtual double                 GetNoiseHyperparam() const   = 0;

        virtual const Eigen::MatrixXd& GetLargeX() const = 0;
        virtual const Eigen::VectorXd& GetSmallY() const = 0;

        Eigen::VectorXd PredictMaximumPointFromData() const;

        Kernel                   GetKernel() const { return m_kernel; }
        KernelThetaDerivative    GetKernelThetaDerivative() const { return m_kernel_theta_derivative; }
        KernelFirstArgDerivative GetKernelFirstArgDerivative() const { return m_kernel_first_arg_derivative; }

    protected:
        Kernel                   m_kernel;
        KernelThetaDerivative    m_kernel_theta_derivative;
        KernelFirstArgDerivative m_kernel_first_arg_derivative;
    };

    // k
    Eigen::VectorXd CalcSmallK(const Eigen::VectorXd& x,
                               const Eigen::MatrixXd& X,
                               const Eigen::VectorXd& kernel_hyperparameters,
                               const Kernel           kernel);

    // K_y = K_f + sigma^{2} I
    Eigen::MatrixXd CalcLargeKY(const Eigen::MatrixXd& X,
                                const Eigen::VectorXd& kernel_hyperparameters,
                                const double           noise_level,
                                const Kernel           kernel);

    // K_f
    Eigen::MatrixXd
    CalcLargeKF(const Eigen::MatrixXd& X, const Eigen::VectorXd& kernel_hyperparameters, const Kernel kernel);

    // partial k / partial x
    Eigen::MatrixXd CalcSmallKSmallXDerivative(const Eigen::VectorXd&         x,
                                               const Eigen::MatrixXd&         X,
                                               const Eigen::VectorXd&         kernel_hyperparameters,
                                               const KernelFirstArgDerivative kernel_first_arg_derivative);

    // partial K_y / partial theta
    std::vector<Eigen::MatrixXd> CalcLargeKYThetaDerivative(const Eigen::MatrixXd&      X,
                                                            const Eigen::VectorXd&      kernel_hyperparameters,
                                                            const KernelThetaDerivative kernel_theta_derivative);

    // partial K_y / partial sigma^{2}
    Eigen::MatrixXd CalcLargeKYNoiseLevelDerivative(const Eigen::MatrixXd& X,
                                                    const Eigen::VectorXd& kernel_hyperparameters,
                                                    const double           noise_level);

} // namespace sequential_line_search

#endif // SEQUENTIAL_LINE_SEARCH_REGRESSOR_HPP
