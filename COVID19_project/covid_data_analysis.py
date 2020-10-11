import pandas as pd 
from matplotlib import pyplot as plt

#TODO: a webservice that download the new files automatically

#read files
confirmed_cases=pd.read_csv("time_series_covid19_confirmed_global.csv")
deaths_cases=pd.read_csv("time_series_covid19_deaths_global.csv")
recovered_cases=pd.read_csv("time_series_covid19_recovered_global.csv")
#choose Ireland and Hungary
confirmed_ie_hu=confirmed_cases[(confirmed_cases["Country/Region"]=="Ireland") | (confirmed_cases["Country/Region"]=="Hungary") ].reset_index(drop=True)
deaths_ie_hu=deaths_cases[(deaths_cases["Country/Region"]=="Ireland") | (deaths_cases["Country/Region"]=="Hungary") ].reset_index(drop=True)
recovered_ie_hu=recovered_cases[(recovered_cases["Country/Region"]=="Ireland") | (recovered_cases["Country/Region"]=="Hungary") ].reset_index(drop=True)

#delete columns
del confirmed_ie_hu["Province/State"]
del confirmed_ie_hu["Lat"]
del confirmed_ie_hu["Long"]

del deaths_ie_hu["Province/State"]
del deaths_ie_hu["Lat"]
del deaths_ie_hu["Long"]

del recovered_ie_hu["Province/State"]
del recovered_ie_hu["Lat"]
del recovered_ie_hu["Long"]

#pivot tables to easier usage
#confirmed
confirmed_ie_hu=confirmed_ie_hu.pivot_table(columns=["Country/Region"]).reset_index()
confirmed_ie_hu.rename(columns = {'index': "Date", "Hungary": "Hungary_C", "Ireland": "Ireland_C"}, inplace=True)
confirmed_ie_hu['Date']=pd.to_datetime(confirmed_ie_hu['Date'],format='%m/%d/%y') 
confirmed_ie_hu.sort_values(by= 'Date').reset_index(inplace=True)


#deaths
deaths_ie_hu=deaths_ie_hu.pivot_table(columns=["Country/Region"]).reset_index(drop=False)
deaths_ie_hu.rename(columns = {'index': "Date", "Hungary": "Hungary_D", "Ireland": "Ireland_D"}, inplace=True)
deaths_ie_hu['Date']=pd.to_datetime(deaths_ie_hu['Date'],format='%m/%d/%y') 
deaths_ie_hu=deaths_ie_hu.sort_values(by= 'Date')
deaths_ie_hu.set_index("Date", inplace = True)

#recovered
recovered_ie_hu=recovered_ie_hu.pivot_table(columns=["Country/Region"]).reset_index(drop=False)
recovered_ie_hu.rename(columns = {'index': "Date", "Hungary": "Hungary_R", "Ireland": "Ireland_R"}, inplace=True)
recovered_ie_hu['Date']=pd.to_datetime(recovered_ie_hu['Date'],format='%m/%d/%y') 
recovered_ie_hu=recovered_ie_hu.sort_values(by= 'Date')
recovered_ie_hu.set_index("Date",  inplace=True)

#merge datasets
all_data=pd.merge(confirmed_ie_hu,deaths_ie_hu, left_on="Date",right_on="Date").merge(recovered_ie_hu, left_on="Date",right_on="Date")
all_data=all_data.sort_values(by="Date", ignore_index=True).reset_index(drop=True)
all_data["Hungary_A"]=all_data["Hungary_C"]-all_data["Hungary_D"]-all_data["Hungary_R"]
all_data["Ireland_A"]=all_data["Ireland_C"]-all_data["Ireland_D"]-all_data["Ireland_R"]


confirmed_hun=all_data[["Hungary_C","Date"]]
confirmed_ie=all_data[["Ireland_C","Date"]]

deaths_hun=all_data[["Hungary_D","Date"]]
deaths_ie=all_data[["Ireland_D","Date"]]

recovered_hun=all_data[["Hungary_R","Date"]]
recovered_ie=all_data[["Ireland_R","Date"]]

active_hun=all_data[["Hungary_A","Date"]]
active_ie=all_data[["Ireland_A","Date"]]

hungarian_rate=100000/10000000
irish_rate=100000/4000000

#confirmed

figure=plt.figure(figsize=(20,20))
plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.subplot(2, 2, 1)
plt.plot(confirmed_hun.Date,confirmed_hun.Hungary_C)
plt.plot(confirmed_ie.Date,confirmed_ie.Ireland_C)
plt.xticks(rotation=45)
plt.title("Hungary and Ireland all confirmed cases")
plt.ylabel("confirmed cases")
plt.legend(["Hungary", "Ireland"])
plt.grid(b=True, alpha=0.5)
#confirmed for 100.000 habitants
plt.subplot(2, 2, 2)
plt.plot(confirmed_hun.Date,confirmed_hun.Hungary_C*hungarian_rate)
plt.plot(confirmed_ie.Date,confirmed_ie.Ireland_C*irish_rate)
plt.xticks(rotation=45)
plt.title("Hungary and Ireland confirmed cases per 100.000 habitants")
plt.ylabel("confirmed cases")
plt.legend(["Hungary", "Ireland"])
plt.grid(b=True, alpha=0.5)

#active cases
plt.subplot(2, 2, 3)
plt.plot(active_hun.Date,active_hun.Hungary_A)
plt.plot(active_ie.Date,active_ie.Ireland_A)
plt.xticks(rotation=45)
plt.title("Hungary and Ireland all active cases")
plt.ylabel("active cases")
plt.legend(["Hungary", "Ireland"])
plt.grid(b=True, alpha=0.5)
#Active cases for 100.000 habitants
plt.subplot(2, 2, 4)
plt.plot(active_hun.Date,active_hun.Hungary_A*hungarian_rate)
plt.plot(active_ie.Date,active_ie.Ireland_A*irish_rate)
plt.xticks(rotation=45)
plt.title("Hungary and Ireland active cases for 100.000 habitants")
plt.ylabel("active cases")
plt.legend(["Hungary", "Ireland"])
plt.grid(b=True, alpha=0.5)
plt.show()
#deaths

figure2=plt.figure(figsize=(20,20))
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.subplot(2, 2, 1)
plt.plot(deaths_hun.Date,deaths_hun.Hungary_D)
plt.plot(deaths_ie.Date,deaths_ie.Ireland_D)
plt.xticks(rotation=45)
plt.title("Hungary and Ireland all deaths")
plt.ylabel("deaths")
plt.legend(["Hungary", "Ireland"])
plt.grid(b=True, alpha=0.5)
#deaths for 100.000 habitants
plt.subplot(2, 2, 2)
plt.plot(deaths_hun.Date,deaths_hun.Hungary_D*hungarian_rate)
plt.plot(deaths_ie.Date,deaths_ie.Ireland_D*irish_rate)
plt.xticks(rotation=45)
plt.title("Hungary and Ireland deaths per 100.000 habitants")
plt.ylabel("deaths")
plt.legend(["Hungary", "Ireland"])
plt.grid(b=True, alpha=0.5)

#recovered
plt.subplot(2, 2, 3)
plt.plot(recovered_hun.Date,recovered_hun.Hungary_R)
plt.plot(recovered_ie.Date,recovered_ie.Ireland_R)
plt.xticks(rotation=45)
plt.title("Hungary and Ireland all recovered")
plt.ylabel("recovered")
plt.legend(["Hungary", "Ireland"])
plt.grid(b=True, alpha=0.5)

plt.subplot(2, 2, 4)
plt.plot(recovered_hun.Date,recovered_hun.Hungary_R*hungarian_rate)
plt.plot(recovered_ie.Date,recovered_ie.Ireland_R*irish_rate)
plt.xticks(rotation=45)
plt.title("Hungary and Ireland recovered per 100.000 habitants")
plt.ylabel("recovered")
plt.legend(["Hungary", "Ireland"])
plt.grid(b=True, alpha=0.5)

plt.show()


#TODO: Write a SIR modell for predicting when will be the daily cases lover than than 25 for 100.000 habitants (EU green list)

