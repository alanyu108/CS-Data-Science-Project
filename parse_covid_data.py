import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.externals._packaging.version import parse
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from numpy.polynomial import Polynomial as P
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline
import os 
from scipy.stats import pearsonr
import datetime

"""
Missing data can affect interpretation of factors that might put people at higher risk for severe disease. 
Analyses of incomplete data elements are likely an underestimate of the true occurrence.
"""

working_dir = os.path.dirname(os.path.realpath(__file__))
nyc_pop = 8804190

#https://data.cityofnewyork.us/Health/COVID-19-Outcomes-by-Testing-Cohorts-Cases-Hospita/cwmx-mvra
# Number of COVID-19 infections in NY
covid_infections = pd.read_csv("covid_cases.csv")
parsed_infections = covid_infections.loc[(covid_infections['specimen_date'].str[6:] == '2020') | (covid_infections['specimen_date'].str[6:] == '2021')]

parsed_infections['Date'] = pd.to_datetime(parsed_infections['specimen_date'], format="%m/%d/%Y")
parsed_infections['Month'] = parsed_infections['Date'].apply(lambda x: x.month)
parsed_infections['Year'] = parsed_infections['Date'].apply(lambda x: x.year)
infect_by_month = parsed_infections.groupby(['Month', 'Year']).agg(
    confirmed = pd.NamedAgg("Number_confirmed",  "mean")
)

#daily infections NYC OpenData
infect_by_day = parsed_infections.groupby(['Date']).agg({"Number_confirmed": "mean"})
infect_by_day['Number_confirmed'] = infect_by_day['Number_confirmed'].apply(lambda x: ( x / nyc_pop) * 100 )



#average infection rate over time between 2020 and 2021
infect_by_month['confirmed'] = infect_by_month['confirmed'].apply(lambda x: ( x / nyc_pop) * 100 )
infect_2020 = infect_by_month.unstack()[('confirmed', 2020)]
infect_2021 = infect_by_month.unstack()[('confirmed', 2021)].fillna(0)
"""
"""



#https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh
covid_vaccine_nyc = pd.read_csv("nyc_vaccine.csv")
covid_vaccine_nyc.dropna(inplace=True)
covid_vaccine_nyc.fillna(0.0, inplace=True)
covid_vaccine_nyc['Date'] = pd.to_datetime(covid_vaccine_nyc['Date'], format="%m/%d/%Y")
vax_per_day = covid_vaccine_nyc.groupby(['Date']).agg({"Series_Complete_Pop_Pct": "mean"})
"""
"""

#vax_per_day joined with infect_by_day (openData NY data set)
joined_data = vax_per_day.merge(infect_by_day,how ="left" ,on="Date")
joined_data.dropna(inplace=True)
joined_data = joined_data[joined_data['Series_Complete_Pop_Pct'] > 0]

vax_rate = joined_data['Series_Complete_Pop_Pct'].to_list()
infection_rate = joined_data['Number_confirmed'].to_list()
"""
"""


"""
#linear regression for data
def mse_cost(pred, y): return np.mean((pred - y) ** 2)
x_train, x_test, y_train, y_test = train_test_split(vax_rate, infection_rate, test_size=0.7);
x_train = np.array(x_train).reshape(-1, 1)
y_train = np.array(y_train).reshape(-1, 1)
x_test = np.array(x_test).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)
clf = LinearRegression()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
score = clf.score(y_test, y_pred)
print(f"The r2 score for the linear model is {score}")
print(f"The slope and y-intercept for the linear model is {clf.coef_[0][0]} and {clf.intercept_[0]}")
print(f"The mean squared error is {mse_cost(x_test, y_pred)}")
plt.plot(x_test, y_pred, label = f"y = {np.round(clf.coef_[0][0], 2)}x + {np.round(clf.intercept_[0], 2)}")

poly = P.fit( x_train, y_train, 2)

print(f"The r2 score is {r2_score(y_test, poly(np.array(x_test, dtype=np.float64)))}")
"""

"""
#bar plot for infection bar graphs
labels = [x for x in range(1, len(infect_2020) + 1)]
data = [
    infect_2020.tolist(),
    infect_2021.tolist()
]
combined_data = list(zip(data[0], labels, [2020 for x in range(len(infect_2020))])) \
                + list(zip(data[1], labels, [2021 for x in range(len(infect_2021))])
)
df = pd.DataFrame(combined_data, columns=["Average_Infections", "month", "year"])

sns.lineplot(x="month",y="Average_Infections", data=df, hue="year" , palette=["red", "blue"])
plt.title("COVID-19 Infection Rate in NYC (2020-2021)")
plt.xlabel("Month")
plt.ylabel("Average infection rate (% population of NYC)")


plt.savefig(working_dir + "/Visuals/line_infect.png", format="png") 
"""


"""
#curve fitting use reciprocal equation
x_train, x_test, y_train, y_test = train_test_split(vax_rate, infection_rate, test_size=0.7)
def func(x, a, b,c):
    return a * (1 / (b + x)) + c;

[a, b, c] , params_covar = curve_fit(func,x_train, y_train)


values = [func(x, a, b, c) for x in x_test];
print(f"the r2 score for this curve fitting is {r2_score(y_test, values)}")
    
y_pred = [func(x, a, b, c) for x in vax_rate];
myline = np.linspace(-1, 60, len(vax_rate))
plt.plot(myline, y_pred, label = f"y = {np.round(a, decimals=2)} * (1 / ({np.round(b, decimals=2)} + x)) +  {np.round(c, decimals=2)}")
sns.scatterplot(x=vax_rate, y=infection_rate, color= "red")
plt.title("Vaccination Rate vs COVID-19 Infection Rate in NYC (2021)")
plt.xlabel("% of New Yorkers Vaccinated")
plt.ylabel("% of New Yorkers Infected")
plt.show()
plt.savefig(working_dir + "\\Visuals\\vax_infect.png", format="png") 
"""


"""

#infection makeup NYC 
#https://data.cdc.gov/Case-Surveillance/COVID-19-Case-Surveillance-Public-Use-Data-with-Ge/n8mc-b4w4
#definition of exposed according to the CDC
# In the 14 days prior to illness onset, did the patient have any of the following known exposures: 
# domestic travel, international travel, cruise ship or vessel travel as a passenger or crew member,
# workplace, airport/airplane, adult congregate living facility (nursing, assisted living,
# or long-term care facility), school/university/childcare center, correctional facility, 
# community event/mass gathering, animal with confirmed or suspected COVID-19,
# other exposure, contact with a known COVID-19 case? [Yes, Unknown, Missing]
nyc_individual = pd.read_csv("nyc_individual_covid_data.csv")
nyc_individual['month'] = nyc_individual['case_month'].apply(lambda x: int(x[5:]))
nyc_individual['year'] = nyc_individual['case_month'].apply(lambda x: int(x[:4]))
nyc_individual['underlying_conditions_yn'].fillna("No", inplace=True)
nyc_individual = pd.get_dummies(nyc_individual, columns=['exposure_yn'], drop_first=True)
month_data = nyc_individual.groupby(['month', 'year']).agg(
    infected = pd.NamedAgg("case_month", aggfunc="count") , # number of people infected
    exposed = pd.NamedAgg("exposure_yn_Yes", aggfunc="sum") #number of people infected due to exposure
).unstack().fillna(0)
month_data['percent_exposed', 2020]= 100 * (month_data['exposed'][2020]  / month_data['infected'][2020])
month_data['percent_exposed', 2021]= 100 * (month_data['exposed'][2021]  / month_data['infected'][2021])
month_data.fillna(0, inplace=True)


percent_exposed_2020 = month_data['percent_exposed', 2020].tolist()
percent_exposed_2021 = month_data['percent_exposed', 2021].tolist()
months = [x for x in range(1, len(percent_exposed_2020) + 1)]

combined_data = list(zip(percent_exposed_2020, months, [2020 for x in range(1, len(percent_exposed_2020) + 1)])) + \
                list(zip(percent_exposed_2021, months, [2021 for x in range(1, len(percent_exposed_2021) + 1)]))

df = pd.DataFrame(combined_data, columns=["percent_infected_exposed", "month", "year"])
sns.lineplot(x="month", y = "percent_infected_exposed", hue = "year", data=df, palette=["red", "blue"])
plt.title("COVID-19 Infection Rate Through Exposure NYC (2020-2021)")
plt.xlabel("Month")
plt.ylabel("Infection rate through exposure (%)")


plt.savefig(working_dir + "/expose_infect.png", format="png") 

"""