# Visiting-home-project

The aim of this project is to compare the statistics of the COVID-19 virus data in Ireland and Hungary. 
In addition to comparing the data, a further aim of the project is to try to predict how the new cases will be changed in these countries based on the previous data. 
The ultimate goal is to use this prediction to calculate when both countries will be on the green list (based on EU regulations), 
so when the daily number of cases will be so low that anybody can travel normally between the two countries without any further costs. 
Or in other words: when would it be worth buying a plane ticket to visit our family.  
 
For this I used the "JHU CSSE COVID-19 Data" url: https://github.com/CSSEGISandData/COVID-19, 
and these 3 articles and codes from towards data science: 
https://towardsdatascience.com/infectious-disease-modelling-part-i-understanding-sir-28d60e29fdfc
https://towardsdatascience.com/infectious-disease-modelling-beyond-the-basic-sir-model-216369c584c4
https://towardsdatascience.com/infectious-disease-modelling-fit-your-model-to-coronavirus-data-2568e672dbc7
 
The documentation of the whole project can be find there: 
 
MAIN STEPS:
 
1. Load the JHU CSSE COVID-19 Data from the public github folder and prepare it for the further analysis
2. Visualize and analyze Hungary and Ireland data using matplotlib pyplot
3. Prepare the SIR model
3. With a SIR model, try to predict the future cases in both countries
4. Used this prediction to calculate the date when these two country will be on the EU Green List
5. Write down the conclusion

