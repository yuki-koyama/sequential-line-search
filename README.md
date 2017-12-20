# Overview
This repository contains a part of the source codes used in our research project on the **sequential line search** method (which is a variant of Bayesian optimization). The core algorithm is in the source codes in the "sequential_line_search_core" folder. This repository also contains the following three example applications:

- **bayesian_optimization_1d_gui**: A visual demonstration of the standard Bayesian optimization applied to a one-dimensional test function. 
- **bayesian_optimization_2d_gui**: A visual demonstration of the standard Bayesian optimization applied to a two-dimensional test function.
- **sequential_line_search_2d_gui**: A visual demonstration of the sequential line search method applied to a two-dimensional test function.

# Project Web Site
http://koyama.xyz/project/sequential_line_search/

# Publication
Yuki Koyama, Issei Sato, Daisuke Sakamoto, and Takeo Igarashi. 2017. Sequential Line Search for Efficient Visual Design Optimization by Crowds. ACM Trans. Graph. 36, 4, pp.48:1--48:11 (2017). (a.k.a. Proceedings of SIGGRAPH 2017)
DOI: https://doi.org/10.1145/3072959.3073598

# Dependencies
- [Qt5](http://doc.qt.io/qt-5/)
- [NLopt](https://nlopt.readthedocs.io/)
- [Eigen](http://eigen.tuxfamily.org/)

# How to Compile and Run
We use [cmake](https://cmake.org/) for managing the source codes. You can compile all the applications by, for example, 
```
mkdir build
cd build
cmake PATH_FOR_THE_SOURCE_DIR
make
```
Then you can test the applications by, for example,
```
./sequential_line_search_2d_gui/SequentialLineSearch2dGui
```

# License
The source codes are distributed under the **MIT License**. For details of the MIT License, see the LICENSE file.

# Contact and Feedback
Yuki Koyama (yuki@koyama.xyz)
