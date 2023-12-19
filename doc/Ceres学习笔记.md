# Ceres学习笔记
## 1.非线性最小二乘问题
 - Ceres可以鲁棒地处理如下形式带边界约束的非线性最小二乘拟合问题：
$$
\min\limits_{x}\frac{1}{2}\sum\limits_{i}\rho_{i}(\left \| f_{i}(x_{i1},\dots,x_{ik} \right \|^{2}),s.t. l_{j} \le x_{j} \le u_{j}
$$
 - 其中$\rho_{i}(\left \| f_{i}(x_{i1},\dots,x_{ik} \right \|^{2})$为残差块(ResidualBlock)，$f_{i}(\cdot)$是一个损失函数(CostFunction)，$[x_{i1},\dots,x_{ik}]$为参数块(ParameterBlock)，$l_{j},u_{j}$是参数块$x_{j}$的边界。
 - 这种形式的问题出现在各个领域：从统计学中的曲线拟合到计算机视觉中根据图像重建出3D模型。

 ## 2.最小值求解问题
 - 首先，考虑求解下面式子最小值：
$$
\frac{1}{2}(10-x)^{2}
$$
 - 这是一个平凡(trival)问题，最小值位于$x=10$。第一步编写一个函数来计算残差$f(x)=10-x$:
```C++
struct CostFunctor {
  template <typename T>
  bool operator()(const T *const x, T *residual) const {
    residual[0] = 10.0 - x[0];
    return true;
  }
};
```
 - 第二步，使用Ceres构造一个非线性最小二乘问题来进行求解：
```C++
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  // The variable to solve for with its initial value.
  double initial_x = 5.0;
  double x = initial_x;

  // Build the problem.
  ceres::Problem problem;

  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).
  ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
  problem.AddResidualBlock(cost_function, nullptr, &x);

  // Run the solver!
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";
  std::cout << "x : " << initial_x
            << " -> " << x << "\n";
  return 0;
}
```
 - AutoDiffCostFunction将CostFunctor作为输入，给予一个CostFunction接口，自动计算其微分。

## 3.微分
 - 和大多数优化工具一样，Ceres求解器也需要能够求解目标函数每一项中任意参数值的微分。
### 3.1 数值微分
 - 一些情况下，不可能定义一个模板损失函数，例如，当估算残差时需要调用一个无法控制的库函数。这种情形下，用户定义一个计算残差值的函数，构造一个数值差分损失函数(NumericDiffCostFunction)，例如$f(x) = 10 - x$，对应函数如下：
```C++
struct NumericDiffCostFunctor {
  bool operator()(const double *const x, double *residual) const {
    residual[0] = 10.0 - x[0];
    return true;
  }
};
```
 - 然后将其添加给$Problem$：
```C++
ceres::CostFunction *cost_function = new ceres::NumericDiffCostFunction<ceres::NumericDiffCostFunctor, ceres::CENTRAL, 1, 1>(new NumericDiffCostFunctor);
problem.AddResidualBlock(cost_function, nullptr, &x);
```
 - 结构和前面使用的自动微分基本一致，除了一个额外的模板参数指示用于计算数值的有限差分类型。
 - 一般来说，我们推荐使用自动微分而不是数值微分。使用 C++ 模板可以提高自动微分效率，而数值微分成本高昂，容易出现数值错误，并导致收敛速度变慢。

### 3.2 解析微分
 - 一些情况下使用不了自动微分，例如，以封闭形式计算微分比依赖使用链式规则的自动微分更有效，这种情况下，可以提供你自己的残差和雅可比计算代码。
 - 如果你在编译阶段就知道参数数目和残差，定义一个CostFunction或者SizedCostFunction的子类：
```C++
class QuadraticCostFunction : public ceres::SizedCostFunction<1, 1> {
public:
  virtual ~QuadraticCostFunction() {}
  virtial bool Evalute(double const *const *parameters, double *residuals, double **jacobians) const {
    const double x = parameters[0][0];
    rediduals[0] = 10 - x;
    if (jacobians != nullptr && jacobians[0] != nullptr) {
      jacobians[0][0] = -1;
    }
    return true;
  }
};
```

## 4.鲍威尔方程(Powell's Function)
 - 最小化鲍威尔方程，令$x=[x_{1}, x_{2}, x_{3}, x_{4}]$：
$$
f_{1}(x) = x_{1} + 10x_{2} \\ f_{2}(x) = \sqrt{5}(x_{3}-x_{4}) \\ f_{3}(x)=(x_{2}-2x_{3})^{2} \\ f_{4}(x)=\sqrt{10}(x_{1} - x_{4})^{2} \\ F(x)=[f_{1}(x), f_{2}(x), f_{3}(x), f_{4}(x)]
$$ 
 - $F(x)$是一个有四个参数的函数，对应四个残差，我们希望找到一个$x$使得$\frac{1}{2}\left \| F(x) \right \|^{2} $最小。
 - 第一步定义函数来评估目标函数的每一个项，对应$f_{4}(x_{1}, x_{4})$：
```C++
struct F4 {
  template <typename T>
  bool operator()(const T *const x1, const T *const x4, T *residual) const {
    redisual[0] = sqrt(10.0) * (x1[0] - x4[0]) * (x1[0] - x4[0]);
    return true;
  }
};
```
 - 同样的可以定义$f_{1}(x_{1}, x_{2}), f_{2}(x_{3}, x_{4}), f_{3}(x_{2}, x_{3})$，问题可以构造成如下：
```C++
double x1 = 3.0, x2 = -1.0, x3 = 0.0, x4 = 1.0;

ceres::Problem problem;
problem.AddResidualBlock(new ceres::AutoDiffCostFunction<F1, 1, 1, 1>(new F1), nullptr, &x1, &x2);
problem.AddResidualBlock(new ceres::AutoDiffCostFunction<F2, 1, 1, 1>(new F2), nullptr, &x3, &x4);
problem.AddResidualBlock(new ceres::AutoDiffCostFunction<F3, 1, 1, 1>(new F3), nullptr, &x2, &x3);
problem.AddResidualBlock(new ceres::AutoDiffCostFunction<F4, 1, 1, 1>(new F4), nullptr, &x1, &x4);
```
 - 最后进行优化：
```C++
ceres::Solver::Options options;
options.max_num_iterations = 100;
options.linear_solver_type = ceres::DENSE_QR;
options.minimizer_progress_to_stdout = true;

ceres::Solver::Summary summary;
ceres::Solve(options, &problem, &summary);
```

## 5.曲线拟合
 - 最小二乘和非线性最小二乘分析的最初目的是拟合曲线。现在考虑一个例子，数据是通过采样曲线$y=e^{0.3x + 0.1}$，并添加标准差为$\sigma = 0.2$的高斯噪声，然后拟合曲线：
$$
y=e^{mx+c}
$$
 - 第一步定义目标模板来估算残差：
```C++
struct ExponentialResidual {
  ExponentialRedisual(double x, double y) : x_(x), y_(y) {}

  template<typename T>
  bool operator()(const T *const m, const T *const c, T *redisual) {
    redidual[0] = y_ - exp(m[0] * x_ + c[0]);
    return true;
  }

private:
  const double x_;
  const double y_;
};
```
 - 假设观测量data是长度为2n大小的数组，问题构造对应简单对每一个观测量创建一个CostFunction：
```C++
double m = 0.0;
double c = 0.0;

ceres::Problem problem;
for (int i = 0; i < kNumObservations; ++i) {
  ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<ExponentialResidual, 1, 1, 1>(new ExponentialResidual(data[2 * i], data[2 * i + 1]));
  problem.AddRedisualBlock(cost_function, nullptr, &m, &c);
}
```
 - 最后，进行优化：
```C++
ceres::Solver::Options options;
options.max_num_iterations = 25;
options.linear_solver_type = ceres::DENSE_QR;
options.minimizer_progress_to_stdout = true;

ceres::Solver::Summary summary;
ceres::Solve(options, &problem, &summary);
```
 - 当存在外点(outliers)时，一个标准技术为损失函数(loss function)，损失函数降低了较大残差对应残差模块，通常对应于异常点。将一个残差块关联一个损失函数，仅仅需要做如下改变：
```C++
problem.AddResidualBlock(cost_function, nullptr , &m, &c);
// TO:
problem.AddResidualBlock(cost_function, new CauchyLoss(0.5) , &m, &c);
```

## 6.光束平差法
 - Ceres主要目标是处理大尺度的光束平差法问题。给定一个观测图像特征位置及对应关系的集合，光束平差法的目标是找到3D点位置和相机参数来最小化重投影误差。优化问题通常表述为一个非线性最小二乘问题，误差为观测特征位置与3D点在相机图像平面上对应投影之间的L2范数平方。
 - 第一步，定义一个模板函数来计算重投影误差/残差。BA问题的每一个残差依赖于一个3D点和一个9参数相机。相机9个参数定义为：3个对应于罗德里格斯旋转轴，3个对应于平移，一个对应于焦距(focal length)，两个对应于径向畸变(radial distortion)。
```C++
struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y) : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T *const camera, const T *const point, T *residuals) const {
    T p[3];
    // camera[0,1,2] are the angle-axis rotation.
    ceres::AngleAxisRotatePoint(camera, point, p);
    // camera[3,4,5] are the translation.
    p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    T xp = -p[0] / p[2];
    T yp = -p[1] / p[2];

    // Apply second and fourth order radial distortion.
    const T &l1 = camera[7];
    const T &l2 = camera[8];
    T r2 = xp * xp + yp * yp;
    T distortion = 1.0 + r2 * (l1 + l2 * r2);

    // Compute final projected point position.
    const T &focal = camera[6];
    T projected_x = focal * distortion * xp;
    T projected_y = focal * distortion * yp;

    // The error is the difference between the predicted and observed position.
    residual[0] = predicted_x - T(observed_x);
    residual[1] = predicted_y - T(observed_y);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from the client code. 
  static ceres::CostFunction *Create(const double observed_x, const double observed_y) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(new SnavelyReprojectionError(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;
}
```
 - 由于这是一个大稀疏问题(对于DENSE_QR太大了)，一个方法是将问题(problem)的线性求解类型(Solver::Options::linear_solver_type)设置为SPARSE_NORMAL_CHOLESKY；与此同时，由于BA问题具有册数的稀疏结构，更合理的做法是找寻一种更加高效的方法，Ceres提供了三个特殊求解器(Schur-based solvers)。
```C++
ceres::Solver::Options options;
options.linear_solver_type = ceres::DENSE_SCHUR;
options.minimizer_progress_to_stdout = true;
ceres::Solver::Summary summary;
ceres::Solve(options, &problem, &summary);
```



