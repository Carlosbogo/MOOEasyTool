# MOO Easy Tool

An user friendly MOO tool

## Main algorithm:

1. Initialize GP model: 

| Attre          | Default-Value             | Description                |
|----------------|---------------------------|----------------------------|
| O              | non-default               | # of objective to optimize |
| C              | non-default               | # of constrains (TBD)      |
| d              | non-default               | input dimension            |
| kernel         | non-default               | kernel of the GP           |
| X              | Empty 2D np array         | input data of the GP       |
| Y              | Empty 2D np array         | output data of the GP      |
| noise_variance | 0.01                      | output noise of the GP     |
| opt            | gpflow.optimizers.Scipy() | Optimizer of GP's kernel   |
| GPR            | None                      | Gaussian Process Regressor |

NOTE1: We need at least 1 sample (x,y) so that GPR is not completly flat without any further assumption.
NOTE2: Minimum noise is 1e-6 which is practically none


2. Get at least 1 random sample (intput sample):
    1. Generate a random intput sample
    2. Evaluate functions and constrains to get its ourput
    3. Add the sample (input, output) to GP model 
    4. Update GPR model
    5. Optimize GPR's kernel hyperparameters

3. For each iteration of the iterations:
    1. Create a searching grid of the input space.
    2. Evaluate acquisition function in the grid to get the optimum.
    3. Add the sample (input, output) to GP model 
    4. Update GPR model
    5. Optimize GPR's kernel hyperparameters


## TODO: Next task to complete

* Correct usage of bounds
* Code and add many acquisition functions
* Implement cmd parameters
* Code efficient benchmark functions and separate them from ain
* Implement constrains
* More efficient search of acquisition function optimum?
* Several samples for each iteration?
* Study to fix 3.4 and 3.5 at the start of step 3
