# MOO Easy Tool

An user friendly MOO tool

## Main algorithm:

1. Initialize GP model: 

| Attr           | Default-Value             | Description                |
|----------------|---------------------------|----------------------------|
| O              | non-default               | # of objective to optimize |
| C              | non-default               | # of constrains (TBD)      |
| d              | non-default               | input space dimensions     |
| kernel         | non-default               | kernel of the GP           |
| X              | Empty 2D np array         | input data of the GP       |
| Y              | Empty 2D np array         | output data of the GP      |
| noise_variance | 0.01                      | output noise of the GP     |
| opt            | gpflow.optimizers.Scipy() | Optimizer of GP's kernel   |
| multiGPR       | None                      | Gaussian Process Regressor |

NOTE1: We need at least 1 sample (x,y) so that GPR is not completly flat without any further assumption.
NOTE2: Minimum noise is 1e-6 which is practically none


2. Get at least 1 random sample (intput sample):
    1. Generate a random intput sample
    2. Evaluate functions and constrains to get its ourput
    3. Add the sample (input, output) to GP model 
    4. Update multiGPR model
    5. Optimize multiGPR's kernel hyperparameters

3. For each iteration of the iterations:
    1. Create a searching grid of the input space.
    2. Evaluate acquisition function in the grid to get the optimum.
    3. Add the sample (input, output) to GP model 
    4. Update multiGPR model
    5. Optimize multiGPR's kernel hyperparameters


## TODO: Next task to complete

### Released:

* ISSUE: GaussianProcess.plotSamples functions for >1 input dimension
* Write output files of the experiments

### Working:

* Implement cmd parameters
* Study to fix 3.4 and 3.5 at the start of step 3
* Final result, pareto front and pareto set

### Planning:

* Code efficient benchmark functions and separate them from main

### Backlog:

* More efficient search of acquisition function optimum?
* Several samples for each iteration?
* Implement constrains
* Improve usage of bounds (each input variable its own bound) 
    * NOTE: Transform input variables instead of bounds?
* Code and add many acquisition functions

* Return values that u have not evaluated as pareto front
