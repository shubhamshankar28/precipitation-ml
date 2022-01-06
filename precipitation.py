import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#print("executing the lengthy part")
train_set = pd.read_csv('precipitation.csv')
## replace > 0 with 1
train_set[(train_set['PRCP'] > 0)] = 1
