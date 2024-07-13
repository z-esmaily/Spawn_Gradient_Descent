# Spawn Gradient Descent
We have provided all the functions, methods under test, and the proposed method in a single file.

The methods being tested include:
- Proposed Method (SpawnGD)
- Gradient Descent
- Gradient Descent with Tuned Step Size
- ADAM
- RADAM
- MOMENTUM
- RMSPROP

To evaluate and display the performance of various methods with a fixed starting point, run the following code:

    python SpawnGD_Fix_Init_Point.py
  
By default, a simple quadratic function is used as the test function, with the initial starting point set to ‍‍‍‍`[0.0, 0.5]`. However, you can select any of the following functions:
- Rosenbrock
- Schaffer
- Ackley
- Matyas
- Stretched Quadratic
- Quadratic(by default)

For the starting point, you should select numbers within the range of `0.0 to 1.0`. 

You can specify these arguments using `--function_name` and `--initial_point` as follows:

    python SpawnGD_Fix_Init_Point.py --function_name <function_name> --initial_point <initial_point>

![Fig_github](https://github.com/user-attachments/assets/f0681ba7-2c3b-4d4d-af37-bdc4542b9e22)
`___`
To evaluate and display the performance of various methods with random starting points over 100 iterations, run the following code:

    python SpawnGD_Random_Init_Point.py
  
Here too, by default, a simple quadratic function is used as the test function. However, you can change this by selecting the desired function name as follows:

    python SpawnGD_Random_Init_Point.py --function_name <function_name>

`___`
For testing on the `**CIFAR` dataset, we based our implmentaation on the existing code from the `[SRSGD]https://github.com/minhtannguyen/SRSGD` method. Our optimizer is defined in the 'spawngd.py' file located in the '*optimizers' folder. 

Experiments were conducted on the '*ResNet' and '*DenseNet' models. The trained models are stored in the '*checkpoint' directory. To compare different methods and view the results, you can run the 'SPAWNGD.ipynb' notebook. 

All of the above materials are available through this '[Google Drive link]https://drive.google.com/drive/folders/1jp--CqS57AgXeLgCOx1HfgFDYy_c7pCo?usp=drive_link'.
