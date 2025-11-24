from roukf import ROUKF, SigmaDistribution
from opencor_helper import SimulationHelper
import numpy as np
import matplotlib.pyplot as plt
import yaml


file_path = 'config.yaml'


with open(file_path, 'r') as file:
        # Load the YAML data into a Python dictionary
        config = yaml.safe_load(file)




def forward_operator(x: np.ndarray, n_states: int, theta: np.ndarray, n_parameters: int) -> int:
    """
    Forward operator A(x, θ) that propagates the state
    Args:
        x: numpy array - State vector (modified in-place)
        n_states: int - Number of states
        theta: numpy array - Parameter vector (modified in-place)
        n_parameters: int - Number of parameters
    Returns:
        int: Status (1 for success, 0 for failure)
    """
    try: 
       
        y.set_param_vals(x_0, x[0:n_states])
        y.set_param_vals(parameters_to_estimate, theta[0:n_parameters])
        y.run()
        x[0:n_states] = y.get_results(x_1)[:,-1]
        y.reset_and_clear()

        return 1
    
    except Exception as e:
        print(f"Error in forward operator: {e}")
        return 0

def observation_operator(x: np.ndarray, n_states: int, z: np.ndarray, n_observations: int) -> None:
    """
    Observation operator H(x) that maps state to observations
    Args:
        x: numpy array - State vector (modified in-place) I don't know 
        n_states: int - Number of states
        z: numpy array - Observation vector (modified in-place)
        n_observations: int - Number of observations
    """
    try:
        
       
        z[0:n_observations] = x[mapping_indices]
      

    except Exception as e:
        print(f"Error in observation operator: {e}")

def data_generator(numSteps):
    timeSeries = np.zeros((1, numSteps))
    y = SimulationHelper(f"{config['File_name']}.cellml", config['max_step_size'], config['Sampling_rate'] * num_steps, maximumNumberofSteps= config['maximumNumberofInteranlSteps'], pre_time=0)
    x_1 = [i  for i in y.simulation.results().states().keys()]
    parameters_to_estimate = [config['Model_name']  + i for i in config['parameters_to_estimate']]
    y.set_param_vals(parameters_to_estimate, config['true_parameters'])
    y.run()
    x =  y.get_results(x_1)
    variables_to_observe = [config['Model_name']  + i for i in config['observations']] 
    mapping_indices  = [x_1.index(item) for item in variables_to_observe]
    
    timeSeries = x[mapping_indices, :]
    # breakpoint()
    
    y.reset_and_clear()
    y.close_simulation()


    return timeSeries

num_steps = config['num_steps']

if config['Verification']:
    observations = data_generator(num_steps)
else:
    observations = np.load(config['Path_to_obs_data'])



y = SimulationHelper(f"{config['File_name']}.cellml", config['max_step_size'], config['Sampling_rate'], maximumNumberofSteps= config['maximumNumberofInteranlSteps'], pre_time=0)
x_0 = [i  for i in y.simulation.results().states().keys()]
x_1 = [i for i in y.simulation.results().states().keys()]

parameters_to_estimate = [config['Model_name']  + i for i in config['parameters_to_estimate']]
variables_to_observe = [config['Model_name']  + i for i in config['observations']] 
mapping_indices  = [x_1.index(item) for item in variables_to_observe]

# Example dimensions
n_observations = config['n_observations']  # Number of observations
n_states = len(y.simulation.results().states().keys())    # Number of states
n_parameters = config['n_parameters']    # Number of parameters to estimate

# Initialize initialGuess and parametersUncertainties
parameters_uncertainty = np.array(config['parameters_uncertainty'], dtype=np.float64)

# initial_guess = np.array(config['initial_state'], dtype=np.float64)
initial_guess = np.zeros((n_states), dtype=np.float64)

for i, state_name in  enumerate(x_0):
    initial_guess[i] = y.data.states()[state_name]

# initial_guess = np.full(n_states, config['initial_state'], dtype=np.float64)
# initial_parameters = np.full(n_parameters, 5, dtype=np.float64)
initial_parameters = np.array(config['initial_parameters'], dtype=np.float64)

# Create uncertainty arrays
states_uncertainty = np.array(config['states_uncertainty'], dtype=np.float64)


# Initialize ROUKF
kalman_filter = ROUKF(
    n_observations,
    n_states,
    n_parameters,
    states_uncertainty,
    parameters_uncertainty,
    SigmaDistribution.SIMPLEX
)

initial_state = np.ascontiguousarray(initial_guess)
initial_parameter = np.ascontiguousarray(initial_parameters)


# Set initial condition
kalman_filter.setState(initial_state)
kalman_filter.setParameters(initial_parameter)
observe = np.zeros((n_observations), dtype=np.float64)


for i in range(num_steps):
    # Execute a step of the Kalman filter
    observe[0:n_observations] = observations[:, i+1]

    error = kalman_filter.executeStep(
        observe,
        forward_operator, 
        observation_operator
    )
    
    state_estimate = kalman_filter.getState()
    params = kalman_filter.getParameters(n_parameters)


    print("###########################")

    print(f"Step_{i}: Error: {error:.6f}, Parameter estimate: {params}")  
    



# Reset the Kalman filter
kalman_filter.reset(
    n_observations,
    n_states,
    n_parameters,
    states_uncertainty,
    parameters_uncertainty,
    SigmaDistribution.SIMPLEX
)
print("Kalman filter has been reset.")

