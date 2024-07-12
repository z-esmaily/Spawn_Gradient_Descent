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

To evaluate and display the performance of various methods with random starting points over 100 iterations, run the following code:

    python SpawnGD_Random_Init_Point.py
  
Here too, by default, a simple quadratic function is used as the test function. However, you can change this by selecting the desired function name as follows:

    python SpawnGD_Random_Init_Point.py --function_name <function_name>
