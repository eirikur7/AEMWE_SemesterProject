from libraries.DOE import DOEGenerator

FILE = "1MKOH_input_parameters.json"

# Initialize the DOE generator with the parameter file
# The export path and parameter path are handled inside the class

doe = DOEGenerator(parameters_filename=FILE)

# Perform different types of sampling
# You can comment/uncomment lines depending on which DOE you want to use

# doe.full_factorial()
# doe.latin_hypercube(samples=50, KOH_concentration=1000.0)
doe.maximin_latin_hypercube(samples=100, iterations=100000, KOH_concentration=1000.0)
# doe.sobol_sampling(samples=50, KOH_concentration=1000.0)

# doe.grid_sampling(levels_per_param=4)
# doe.fractional_factorial(base_design="a b c ab ac bc abc")
