# Overview
This repository contains a part of the source codes used in our research project on the **sequential line search** method (which is a variant of Bayesian optimization). The core algorithm is in the source codes in the "mail" folder. This repository also contains the following three example applications:
- bayesian_optimization_1d_gui: A visual demonstration of the standard Bayesian optimization applied to a one-dimensional test function. 
- bayesian_optimization_2d_gui: A visual demonstration of the standard Bayesian optimization applied to a two-dimensional test function.
- sequential_line_search_2d_gui: A visual demonstration of the sequential line search method applied to a two-dimensional test function.

# Project Site
http://koyama.xyz/project/sequential_line_search/

# Publication
Yuki Koyama, Issei Sato, Daisuke Sakamoto, and Takeo Igarashi. 2017. Sequential Line Search for Efficient Visual Design Optimization by Crowds. ACM Trans. Graph. 36, 4, pp.48:1--48:11 (2017). (a.k.a. Proceedings of SIGGRAPH 2017)

# Requirements
The source codes have been written for and tested on macOS with x86 architecture only. If you want to use different platforms, you may need to tweak some parts on your own. [Qt](https://www.qt.io/) should be pre-installed for qmake and GUI widgets.

# License
The source codes (except for those explicitly described as third party codes) are distributed under the **MIT License**. For details of the MIT License, see the LICENSE file.

# Third Parties
- [Eigen](http://eigen.tuxfamily.org/) is provided under MPL2.
- [NLopt](https://nlopt.readthedocs.io/) is provided under LGPL.

# Contact and Feedback
Yuki Koyama (yuki@koyama.xyz)
