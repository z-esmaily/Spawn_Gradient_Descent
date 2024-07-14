# Spawn Gradient Descent
*Official Python implementation of the manuscript "Spawning Gradient Descent (SpGD): A Novel Optimization Algorithm for Enhanced Deep Neural Networks" by Moeinoddin Sheikhottayefe, Zahra Esmaily, and Fereshte Dehghani.*

## Overview
SpGD enhances traditional gradient-based optimization techniques by optimizing movement patterns, selecting appropriate starting points, and dynamically adjusting the learning rate using the Augmented Gradient Descent (AGD) algorithm. These innovations reduce zigzagging, improve initial positioning, and eliminate the need for manual learning rate tuning. SpGD incorporates controlled randomization to address limitations of traditional methods, outperforming existing techniques in execution time and accuracy. It is particularly suitable for training deep learning models, achieving improved accuracy quickly and demonstrating efficient convergence with high accuracy in a relatively short number of iterations.

----
## Provided Methods
The implementation includes the following optimization methods:
- Proposed Method (SpGD)
- Gradient Descent
- Gradient Descent with Tuned Step Size
- ADAM
- RADAM
- MOMENTUM
- RMSPROP

## Fixed Starting Point Evaluation

To evaluate and display the performance of various methods with a fixed starting point, run the following code:

    python SpawnGD_Fix_Init_Point.py
  
By default, a simple quadratic function is used as the test function, with the initial starting point set to ‍‍‍‍`[0.0, 0.5]`. However, you can select any of the following functions:
- Rosenbrock
- Schaffer
- Ackley
- Matyas
- Stretched Quadratic
- Quadratic(default)

For the starting point, you should select numbers within the range of `0.0 to 1.0`. 

You can specify these arguments using `--function_name` and `--initial_point` as follows:

    python SpawnGD_Fix_Init_Point.py --function_name <function_name> --initial_point <initial_point>

![Fig_github](https://github.com/user-attachments/assets/f0681ba7-2c3b-4d4d-af37-bdc4542b9e22)
  Figure: Plot of points obtained by various methods on the quadratic function with a fixed starting point of [0.0, 0.5] over 27 steps.

## Random Starting Point Evaluation

To evaluate and display the performance of various methods with random starting points over 100 iterations, run the following code:

    python SpawnGD_Random_Init_Point.py
  
Here too, by default, a simple quadratic function is used as the test function. However, you can change this by selecting the desired function name as follows:

    python SpawnGD_Random_Init_Point.py --function_name <function_name>

----
## CIFAR Dataset Testing

For testing on the **CIFAR** dataset, we based our implementation on the existing code from the [SRSGD](https://github.com/minhtannguyen/SRSGD) method. Our optimizer is defined in the `spawngd.py` file located in the *optimizers* folder. 

Experiments were conducted on the *ResNet* and *DenseNet* models. The trained models are stored in the *checkpoint* directory. To compare different methods and view the results, you can run the `SPAWNGD.ipynb` notebook. 

All materials related to the CIFAR dataset testing are available through this [Google Drive link](https://drive.google.com/drive/folders/1jp--CqS57AgXeLgCOx1HfgFDYy_c7pCo?usp=drive_link).
