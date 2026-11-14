"""Fixed settings. Everything that could be tuned after seeing results is set here, once."""

SEED = 42
ALPHA = 0.05
POWER = 0.80

# Designed allocation of the Criteo experiment: about 85% treated, 15% control (control first).
DESIGN_RATIO_CONTROL_TREATMENT = (15.0, 85.0)

# ASSUMPTION, not data: the smallest lift in visit rate (absolute, 0.005 = 0.5 percentage
# points) that would justify running the campaign. The Criteo file has no cost or revenue
# information, so this number is a placeholder for a business input. The memo shows how the
# decision changes for other values.
PRACTICAL_THRESHOLD = 0.005
THRESHOLD_SENSITIVITY = (0.0025, 0.005, 0.0075, 0.010)

# Simulation sizes for the checks of the toolkit itself.
AA_REPS = 2000
AA_N_PER_ARM = 20_000
POWER_REPS = 2000
DECISION_REPS = 1000
DECISION_N_MULTIPLIERS = (1, 4, 16)

# Uplift evaluation.
BOOTSTRAP_REPS = 200
QINI_BINS = 100
CAPTURE_FRACTIONS = (0.1, 0.3)
CAUSAL_FOREST_TRAIN_N = 200_000
CAUSAL_FOREST_EVAL_N = 500_000
