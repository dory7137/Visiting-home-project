import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#%matplotlib inline 
import mpld3
#mpld3.enable_notebook()
from scipy.integrate import odeint
import lmfit
from lmfit.lineshapes import gaussian, lorentzian
import warnings
warnings.filterwarnings('ignore')



#read files
confirmed_cases=pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
deaths_cases=pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
recovered_cases=pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")
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

#TODO: check null data, celaning


#pivot tables to easier usage
#confirmed
confirmed_ie_hu=confirmed_ie_hu.pivot_table(columns=["Country/Region"]).reset_index()
confirmed_ie_hu.rename(columns = {'index': "Date", "Hungary": "Hungary_C", "Ireland": "Ireland_C"}, inplace=True)
confirmed_ie_hu['Date']=pd.to_datetime(confirmed_ie_hu['Date'],format='%m/%d/%y') 
confirmed_ie_hu.sort_values(by= 'Date').reset_index(inplace=True)

#TODO: use a function

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

hungarian_rate=100000/9773000
irish_rate=100000/4953000

def data_plotter(plt_type, splot_size, xaxis_data, first_dataset, second_dataset, fig_title, fig_ylabel, fig_legend):
    
    if plt_type == "cumulate":
        plt.subplot(splot_size[0], splot_size[1], splot_size[2])
        plt.plot(xaxis_data,first_dataset)
        plt.plot(xaxis_data,second_dataset)
        plt.xticks(rotation=45)
        plt.title(fig_title)
        plt.ylabel(fig_ylabel)
        plt.legend(fig_legend)
        plt.grid(b=True, alpha=0.5)
        #for 100.000 habitants
        plt.subplot(splot_size[0], splot_size[1], splot_size[2]+1)
        plt.plot(xaxis_data,first_dataset*hungarian_rate)
        plt.plot(xaxis_data,second_dataset*irish_rate)
        plt.xticks(rotation=45)
        plt.title(fig_title+" for 100.000 habitants")
        plt.ylabel(fig_ylabel)
        plt.legend(fig_legend)
        plt.grid(b=True, alpha=0.5)
    
    elif plt_type == "daily":
        plt.subplot(splot_size[0], splot_size[1], splot_size[2])
        plt.bar(xaxis_data, first_dataset, width=0.5)
        plt.bar(xaxis_data + pd.Timedelta(hours=12), second_dataset, width=0.5)
        plt.title(fig_title)
        plt.ylabel(fig_ylabel)
        plt.legend(fig_legend)
        plt.grid(b=True, alpha=0.5)
        plt.xticks(rotation=45)
        plt.ylim( ymin = 0)
    else:
        #TODO: raise error
        pass


#confirmed

figure=plt.figure(figsize=(20,20))
plt.subplots_adjust(wspace=0.2, hspace=0.5)
#first plot
#confirmed cases
data_plotter("cumulate",[2,2,1],all_data.Date, all_data.Hungary_C, all_data.Ireland_C, "Hungary and Ireland all confirmed cases", "Confirmed cases", ["Hungary", "Ireland"] )
#active cases
data_plotter("cumulate",[2,2,3],all_data.Date, all_data.Hungary_A, all_data.Ireland_A, "Hungary and Ireland all active cases", "Active cases", ["Hungary", "Ireland"] )
plt.show()

#plot2
figure2=plt.figure(figsize=(20,20))
plt.subplots_adjust(wspace=0.2, hspace=0.5)
#deaths
data_plotter("cumulate",[2,2,1],all_data.Date, all_data.Hungary_D, all_data.Ireland_D, "Hungary and Ireland all deaths", "Deaths", ["Hungary", "Ireland"] )
#recovered
data_plotter("cumulate",[2,2,3],all_data.Date, all_data.Hungary_R, all_data.Ireland_R, "Hungary and Ireland all deaths", "Deaths", ["Hungary", "Ireland"] )
plt.show()

#calculate daily data

daily_confirmed_hun =[]
daily_confirmed_ie=[]
daily_deaths_hun=[]
daily_deaths_ie=[]
daily_recovered_hun=[]
daily_recovered_ie=[]
daily_active_hun=[]
daily_active_ie=[]

# choose the data for figures

for i in range(len(all_data[["Hungary_C"]])):
    if i == 0:
        daily_confirmed_hun.append(all_data[["Hungary_C"]].iat[i,0])
        daily_confirmed_ie.append(all_data[["Ireland_C"]].iat[i, 0])
        daily_deaths_hun.append(all_data[["Hungary_D"]].iat[i, 0])
        daily_deaths_ie.append(all_data[["Ireland_D"]].iat[i, 0])
        daily_recovered_hun.append(all_data[["Hungary_R"]].iat[i, 0])
        daily_recovered_ie.append(all_data[["Ireland_R"]].iat[i, 0])
        daily_active_hun.append(all_data[["Hungary_C"]].iat[i, 0]-all_data[["Hungary_D"]].iat[i, 0]-all_data[["Hungary_R"]].iat[i, 0])
        daily_active_ie.append(all_data[["Ireland_C"]].iat[i, 0]-all_data[["Ireland_D"]].iat[i, 0]-all_data[["Ireland_R"]].iat[i, 0])

    else:
        daily_confirmed_hun.append(all_data[["Hungary_C"]].iat[i, 0] - all_data[["Hungary_C"]].iat[i-1,0])
        daily_confirmed_ie.append(all_data[["Ireland_C"]].iat[i, 0] - all_data[["Ireland_C"]].iat[i-1,0])
        daily_deaths_hun.append(all_data[["Hungary_D"]].iat[i, 0] - all_data[["Hungary_D"]].iat[i-1,0])
        daily_deaths_ie.append(all_data[["Ireland_D"]].iat[i, 0] - all_data[["Ireland_D"]].iat[i-1,0])
        daily_recovered_hun.append(all_data[["Hungary_R"]].iat[i, 0] - all_data[["Hungary_R"]].iat[i-1,0])
        daily_recovered_ie.append(all_data[["Ireland_R"]].iat[i, 0] - all_data[["Ireland_R"]].iat[i-1,0])
        hu_act=(all_data[["Hungary_C"]].iat[i, 0] - all_data[["Hungary_C"]].iat[i-1,0])-(all_data[["Hungary_D"]].iat[i, 0] - all_data[["Hungary_D"]].iat[i-1,0])- (all_data[["Hungary_R"]].iat[i, 0] - all_data[["Hungary_R"]].iat[i-1,0])
        ie_act=(all_data[["Ireland_C"]].iat[i, 0] - all_data[["Ireland_C"]].iat[i-1,0])-(all_data[["Ireland_D"]].iat[i, 0] - all_data[["Ireland_D"]].iat[i-1,0])- (all_data[["Ireland_R"]].iat[i, 0] - all_data[["Ireland_R"]].iat[i-1,0])
        daily_active_hun.append(hu_act)
        daily_active_ie.append(ie_act)


figure3=plt.figure(figsize=(60,20))
plt.subplots_adjust(wspace=0.2, hspace=0.5)

data_plotter("daily",[2,2,1],all_data.Date, daily_confirmed_hun, daily_confirmed_ie, "Hungary and Ireland daily confirmed cases", "daily confirmed cases", ["Hungary", "Ireland"] )
data_plotter("daily",[2,2,2],all_data.Date, daily_active_hun, daily_active_ie, "Hungary and Ireland daily active cases", "daily active cases", ["Hungary", "Ireland"] )
data_plotter("daily",[2,2,3],all_data.Date, daily_deaths_hun, daily_deaths_ie, "Hungary and Ireland daily deaths", "daily deaths", ["Hungary", "Ireland"] )
data_plotter("daily",[2,2,4],all_data.Date, daily_recovered_hun, daily_recovered_ie, "Hungary and Ireland daily recovered", "daily recovered", ["Hungary", "Ireland"] )
plt.show()

daily_confirmed_hun_100K = [case*hungarian_rate for case in daily_confirmed_hun]
daily_confirmed_ie_100K = [case*irish_rate for case in daily_confirmed_ie]
daily_deaths_hun_100K = [case*hungarian_rate for case in daily_deaths_hun]
daily_deaths_ie_100K = [case*irish_rate for case in daily_deaths_ie]
daily_recovered_hun_100K = [case*hungarian_rate for case in daily_recovered_hun]
daily_recovered_ie_100K = [case*irish_rate for case in daily_recovered_ie]
daily_active_hun_100K = [case*hungarian_rate for case in daily_active_hun]
daily_active_ie_100K = [case*irish_rate for case in daily_active_ie]

figure4=plt.figure(figsize=(60,20))
plt.subplots_adjust(wspace=0.2, hspace=0.5)

data_plotter("daily",[2,2,1],all_data.Date, daily_confirmed_hun_100K, daily_confirmed_ie_100K, "Hungary and Ireland daily confirmed cases for 100.000 habitants", "daily confirmed cases", ["Hungary", "Ireland"] )
data_plotter("daily",[2,2,2],all_data.Date, daily_active_hun_100K, daily_active_ie_100K, "Hungary and Ireland daily active cases for 100.000 habitants", "daily active cases", ["Hungary", "Ireland"] )
data_plotter("daily",[2,2,3],all_data.Date, daily_deaths_hun_100K, daily_deaths_ie_100K, "Hungary and Ireland daily deaths for 100.000 habitants", "daily deaths", ["Hungary", "Ireland"] )
data_plotter("daily",[2,2,4],all_data.Date, daily_recovered_hun_100K, daily_recovered_ie_100K, "Hungary and Ireland daily recovered for 100.000 habitants", "daily recovered", ["Hungary", "Ireland"] )
plt.show()

# Write a SIR modell for predicting when will be the daily cases lover than than 25 for 100.000 habitants (EU green list)
#SIR model because in this case the difference between E and I states doesn't count I use the SIR model instead of SEIR 
# and I don't calculate with the ICU beds and invasive ventillators to simplify the solution 

# gamma: how many days takes to recover
gamma = 1.0/9.0
# phi how many days takes to die
phi = 1.0/18.0
# alpha is the fatality rate
alpha=2.3/10.0
N_hungary=9700000
N_ireland=4700000
hungarian_green_list_limit=25.0 / hungarian_rate
irish_green_list_limit=25.0 / irish_rate


 
def data_fitting_SIR(country,N_country, green_list_limit, data_to_fit, outbreak_day, params_to_fit):
    def deriv(y, t, beta, gamma, phi, alpha, N):
        S, I, R, D = y
        dSdt = -beta(t) * I * S / N
        dIdt = beta(t) * I * S/N - (gamma*(1-alpha)*I) - (phi*alpha*I)
        dRdt = gamma * alpha * I
        dDdt = phi * (1-alpha)*I
        return dSdt, dIdt,  dRdt, dDdt

    def logistic_R_0(t, R_0_start, k, x0, R_0_end):
        return (R_0_start-R_0_end) / (1 + np.exp(-k*(-t+x0))) + R_0_end
    
    def Model(days, N, R_0_start, k, x0, R_0_end):
        def beta(t):
            return logistic_R_0(t, R_0_start, k, x0, R_0_end) * gamma

        y0 = N-1.0, 1.0, 0.0, 0.0
        t = np.linspace(0, days-1, days)
        ret = odeint(deriv, y0, t, args=(beta, gamma, phi, alpha, N ))
        S, I, R, D = ret.T
        R_0_over_time = [beta(i)/gamma for i in range(len(t))]

        return t, S, I, R, D, R_0_over_time

    def plotter(t, S, I, R, D, R_0, x_ticks=None, country=""):
    
        f, ax = plt.subplots(1,1,figsize=(20,4))
        if x_ticks is None:
            ax.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infected')
            #ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
            #ax.plot(t, D, 'k', alpha=0.7, linewidth=2, label='Dead')
        else:
            ax.plot(x_ticks, I, 'r', alpha=0.7, linewidth=2, label='Infected')
            #ax.plot(x_ticks, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
            #ax.plot(x_ticks, D, 'k', alpha=0.7, linewidth=2, label='Dead')

            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
            f.autofmt_xdate()

        
        ax.title.set_text('The SIR-Model prediction of '+ country)
        
        ax.grid(b=True, alpha=0.5)
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)

        plt.show()
    #fitting the model
    def fitter(x, R_0_start, k, x0, R_0_end):
        #Model(days, N, R_0_start, k, x0, R_0_end)
        ret = Model(days, N_country, R_0_start, k, x0, R_0_end)
        # index=2 -> returns I
        return ret[2][x]

    fitting_data = data_to_fit
    
    outbreak_shift = outbreak_day
    #form: {parameter: (initial guess, minimum value, max value)}
    params_init_min_max = params_to_fit  

    days = outbreak_shift + len(fitting_data)
    if outbreak_shift >= 0:
        y_data = np.concatenate((np.zeros(outbreak_shift), fitting_data))
    else:
        y_data = y_data[-outbreak_shift:]

    x_data = np.linspace(0, days - 1, days, dtype=int)  # x_data is just [0, 1, ..., max_days] array

    mod = lmfit.Model(fitter)

    for kwarg, (init, mini, maxi) in params_init_min_max.items():
        mod.set_param_hint(str(kwarg), value=init, min=mini, max=maxi, vary=True)

    params = mod.make_params()
    fit_method = "leastsq"

    result = mod.fit(y_data, params, method="least_squares", x=x_data)
    #print(result.best_values)
    result.plot_fit(datafmt="-")
    plt.title("Fitting the model to {}'s data".format(country))
    plt.grid(b=True, alpha=0.5)
    plt.show()

    full_days = 650
    first_date = np.datetime64(all_data.Date.min()) - np.timedelta64(outbreak_shift,'D')
    x_ticks = pd.date_range(start=first_date, periods=full_days, freq="D")

    plotter(*Model(full_days, N_country, **result.best_values), x_ticks=x_ticks, country=country)

    prediction_country = Model(full_days, N_country, **result.best_values)
    active_cases_pred=prediction_country[2]

    #calculate the daily new cases
    daily_prediction_country=[]

    for i in range(len(active_cases_pred)):
        if i == 0:
            daily_prediction_country.append(active_cases_pred[i])
        else:
            daily_prediction_country.append(active_cases_pred[i]-active_cases_pred[i-1])

    

    #calculate the "green list point"
    count=daily_prediction_country.index(max(daily_prediction_country)) #after max

    #find the first element that is above of the limit after the peak
    while(daily_prediction_country[count] > green_list_limit and count <= len(daily_prediction_country)):
        count+=1

    green_list_limit_index=count

    #figure_prob, ax_prob =plt.subplots(1,1 figsize=(20,20))
    figure_prob=plt.figure(figsize=(20,20))
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    ax_prob=plt.subplot(2,1,1)
    plt.hlines(y = green_list_limit, xmin = x_ticks[0], xmax = x_ticks[-2], color='g')
    plt.bar(x_ticks, daily_prediction_country, width=0.5)
    plt.title("{} predicted daily new cases".format(country))
    plt.ylabel("daily new cases")
    ax_prob.xaxis.set_major_locator(mdates.MonthLocator())
    ax_prob.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax_prob.xaxis.set_minor_locator(mdates.MonthLocator())
    figure_prob.autofmt_xdate()
    plt.grid(b=True, alpha=0.5)
    plt.xticks(rotation=45)
    plt.ylim( ymin = 0)
    plt.xlim(x_ticks[40], x_ticks[500])

    ##zoom in
    #figure_zoom, ax_zoom =plt.subplots(1,1 figsize=(20,20))
    ax_zoom=plt.subplot(2,1,2)
    plt.hlines(y = green_list_limit, xmin = x_ticks[0], xmax = x_ticks[-2], color='g')
    plt.bar(x_ticks, daily_prediction_country, width=0.5)
    plt.title("{} predicted daily inflected cases".format(country))
    plt.ylabel("daily inflected cases")
    ax_zoom.xaxis.set_major_locator(mdates.DayLocator())
    ax_zoom.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.grid(b=True, alpha=0.5)
    plt.xticks(rotation=45)
    plt.ylim( ymin = 0, ymax=green_list_limit+500)
    plt.xlim(x_ticks[green_list_limit_index-3], x_ticks[green_list_limit_index+10])
    plt.show()

data_fitting_SIR ("Hungary",N_hungary,hungarian_green_list_limit, all_data.Hungary_C, 42, {"R_0_start": (1.2, 1.0, 2.0), "k": (20.0, 1.0, 100.0), "x0": (150, 100, 400), "R_0_end": (0.8, 0.5, 2.0)})
data_fitting_SIR ("Ireland",N_ireland,irish_green_list_limit, all_data.Ireland_C, 0, {"R_0_start": (1.5, 1.2, 2.0), "k": (40.0, 1.0, 100.0), "x0": (270, 265, 400), "R_0_end": (1.1, 0.98, 2.0)})

#TODO: write down the final conclusion
#TODO: finish the documentation in the jupyter notebook