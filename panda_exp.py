import pandas as pd
from collections import OrderedDict
from datetime import date

sales = {'account': ['Jones LLC', 'Alpha Co', 'Blue Inc'],
         'Jan': [150, 200, 50],
         'Feb': [200, 210, 90],
         'Mar': [140, 215, 95]}



df = pd.DataFrame.from_dict(sales)
df
print(df)

F_cost = [80,210,34,50,30,0]
R_cost = [240,12,40,30,60,40]
Criterion = ["length", "annealling temp", "cg_content", "specificity", "runs", "repeats"]

# Condition_Met = [False, False, False, False, False, False]

F_Primer = {"Cost": F_cost, "Criterion": Criterion, "Condition Met?": [x == 0 for x in F_cost]}
F_table = pd.DataFrame.from_dict(F_Primer)
F_table = F_table[["Criterion", "Cost", "Condition Met?"]]
print(F_table)
