
# coding: utf-8

# # Insights of World Population data for year 2007 w.r.t GDP
# 
# 
# 

# ### Import required library and define constants

# In[199]:

import pandas as pd
import os
import math
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# ### Files

# In[200]:

globalPopulation = pd.read_csv('DataSets/gapminder.csv')

Fileshape = globalPopulation.shape
Filesize_KB = os.path.getsize('DataSets/gapminder.csv')/1024

print(Fileshape)
print("gapminder.csv: " + str(round(Filesize_KB,2)) + " KB")


# ### Sneak Peak of data

# In[201]:

globalPopulation_df = pd.DataFrame(globalPopulation)
globalPopulation_df.head()


# In[202]:

pop = globalPopulation_df['population']
np_pop = np.array(pop)/500000
np_cont = np.array(globalPopulation_df['cont'])

col_dict = {
    'Asia':'red',
    'Europe':'green',
    'Africa':'blue',
    'Americas':'yellow',
    'Oceania':'black'
}
col = [col_dict[key] for key in np_cont]


# ### World Development for 2007

# In[203]:

plt.scatter(x = globalPopulation_df['gdp_cap'], y = globalPopulation_df['life_exp'], s = np_pop, c = col, alpha = 0.8) 

plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('India v/s China Development in 2007')
plt.xticks([1000,10000,100000], ['1k','10k','100k'])

plt.text(1550, 71, 'India', fontsize=20)
plt.text(5700, 80, 'China', fontsize=20)

plt.grid(True)

plt.show()
print(col_dict)


# In[204]:

continent_population = {}
continent_gdp = {}
continent = globalPopulation_df['cont']
for index, row in globalPopulation_df.iterrows():
    entry = row['cont']
    if entry in continent_population.keys():
        continent_population[entry] = int((continent_population[entry] + row['population'])/2)
        continent_gdp[entry] = (continent_gdp[entry] + row['gdp_cap'])/2
    else:
        continent_population[entry] = int(row['population'])
        continent_gdp[entry] = row['gdp_cap']

print("\ncontinent_population: "+ str(continent_population) + "\n")
print("continent_gdp: " + str(continent_gdp) +"\n")


# #### Average world population in 2007 continent-wise

# In[207]:

plt.bar(range(len(continent_population)), continent_population.values(), align='center', color = 'green')
plt.xticks(range(len(continent_population)), list(continent_population.keys()))
plt.xlabel('Continents')
plt.ylabel('Average Population [x 10 million]')
plt.title('Average world population in 2007 continent-wise')
plt.show()


# #### Average world GDP continent-wise

# In[208]:

plt.bar(range(len(continent_gdp)), continent_gdp.values(), align='center')
plt.xticks(range(len(continent_gdp)), list(continent_gdp.keys()))
plt.xlabel('Continents')
plt.ylabel('Average GDP per capita [in USD]')
plt.title('Average world GDP in 2007 continent-wise')
plt.show()

