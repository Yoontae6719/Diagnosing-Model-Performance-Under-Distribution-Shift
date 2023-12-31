# Diagnosing Model Performance Under Distribution Shift

This repository contains a reproduction of the research [Diagnosing Model Performance Under Distribution Shift](https://arxiv.org/abs/2303.02011). Please note that this is not an official repository maintained by the original authors.



## Overviwe
In this reproduction, we aim to recreate the experiments and results reported in the original research paper. We follow the methodology described in the paper as closely as possible, using the similar dataset, parameters, and evaluation metrics. *The data used in the following code is different from the data used in the paper, so the results may not be completely identical. As the official code has not been released yet, please report any incorrect parts*

The repository contains code for reproducing the experiments in the paper. All reproducible experiments are available in the form of Jupyter notebooks. Specifically, it includes:
- [x] 4.1.1 Y | X shift: missing/unobserved covariates
- [x] 4.1.2 X shift: selection bias in age
- [x] Algorithm 1 (See main.py)



## Contributing
We welcome contributions to this repository, such as bug fixes or enhancements to the code. Please create a pull request with your changes, and we will review them as soon as possible.


## Contact
If you have any questions or want to use the code, please contact `yoontae@unist.ac.kr`.


## Aknoldegement
We would like to express our gratitude to the authors of the original research paper for their important contributions to the field.

The original research paper was authored by:
- Tiffany (Tianhui) Cai, Department of Statistics, Columbia University
- Hongseok Namkoong, Department of Decision, Risk, and Operations Division, Columbia University
- Steve Yadlowsky, Brain Team, Google Research

## Related work
In time series (regression) tasks, there are similar studies on the topic mentioned above. If you're interested, please check them out.
- [x] [RevIN](https://openreview.net/forum?id=cGDAkQo1C0p), ICLR 2022
- [x] [Dish-TS](https://arxiv.org/abs/2302.14829), AAAI 2023
- [x] [Stable learning](https://arxiv.org/abs/1911.12580), AAAI 2020
- [x] [TGN](https://openreview.net/pdf?id=sWOsRj4nT1n), ICLR 2023 
