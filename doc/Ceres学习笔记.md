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



