# IMPORTS #
from classes.converters.arcma import ARCMAConverter
from classes.commons.classification_functions import load_training_data_with_window_from_sql
import matplotlib.pyplot as plt

arcma = ARCMAConverter("..\\..\\\\databases\\arcma")
training, test = load_training_data_with_window_from_sql(arcma, "..\\..\\arcma.db", "Select x, y from arcma where person = 1 and activity = 4", "activity", 5000)
x = training[0]["x"]
y = training[0]["y"]
t = range(0, len(x))
plt.plot(t, x)
plt.show()