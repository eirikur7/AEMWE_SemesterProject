from libraries.DOE import DOEGenerator

FILE = "1MKOH_input_parameters.json"

# The path to the JSON file containing the parameters is set in the DOEGenerator class.
# The path to export the results is also set in the DOEGenerator class.
doe = DOEGenerator(parameters_filename=FILE)


doe.full_factorial()
# doe.latin_hypercube(samples=2, KOH_concentration=1000.0)
# doe.random_sampling(samples=50)
# doe.g
# rid_sampling(levels_per_param=4)
# doe.fractional_factorial(base_design="a b c ab ac bc abc")
