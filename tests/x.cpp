#include <iostream>
#include <Eigen/Dense>
 
int main()
{
  Eigen::Matrix2d mat;
  mat << 1, 2,
         3, 4;
//   Eigen::Vector2d u(-1,1), v(2,0);



    std::cout << Eigen::VectorXd::Ones(5).norm() << std::endl;
}