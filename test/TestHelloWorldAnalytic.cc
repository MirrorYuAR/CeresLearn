#include <vector>
#include <ceres/ceres.h>
#include <glog/logging.h>

// number of residuals
// size of first parameters
class QuadraticCostFunction : public ceres::SizedCostFunction<1, 1> {
public:
  bool Evaluate(double const *const *parameters, double *redisuals, double **jacobians) const override {
    double x = parameters[0][0];
    // f(x) = 10 - x;
    redisuals[0] = 10 - x;

    // f'(x) = -1
    if (jacobians != nullptr && jacobians[0] != nullptr) {
      jacobians[0][0] = -1;
    }
    return true;
  }
};

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  double x = 0.5;
  const double initial_x = x;

  ceres::Problem problem;
  ceres::CostFunction *cost_function = new QuadraticCostFunction;
  problem.AddResidualBlock(cost_function, nullptr, &x);

  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";
  std::cout << "x: " << initial_x << "->" << x << "\n";

  return 0;
} 
