# **Title: Causation between COVID-19 vaccinations and infection rates** 
## **Objective**: 
    Using publicly available data provided by the CDC and NY State, we will evaluate if there is causation between COVID-19 vaccines and infections rates. As often said in statistics, correlation does not always mean causation, so we will also be evaluating other factors that can impact infection rates within this project.
## **Background**: 
    - With the recent NYC vaccine mandate and other similar mandates issued across the country, understanding whether the vaccine has an impact on COVID-19 infections is important to understand before we open ourselves back up to the world. 

    - With this COVID-19 vaccine being one of the quickest vaccine to be developed and one of the first mRNA vaccines, it is understandable why so many people are skeptical. 

    - Some critics have stated that the vaccine is just unnecessary considering how low the death rate of the virus is while others have pointed out that it is not the death rate that is the issue but rather the toll the virus can have on our healthcare system. 

    - As both sides would agree on, it is important to do "your own research" so in this case, we will be taking publicly available data and with the power of statistics, transforming said data into something we can understand.

    - In this case, we would be using a Jupyter notebook and going step by step in creating our charts and maps. Each step will have an explanation of what is happening as well as explaining why we choosing to use the statistical tools that we are using.

    - By the end of this project, we should arrive to some sort of conclusion to our problem using a format and visualization that most people would understand.
    


## **Solution Overview**: 
### Dependencies and assumptions: 
1. Python v3.96
2. Numpy v1.21.2
3. Pandas v1.3.2
4. Matplotlib v3.4.3
### **Inputs**: 
1. [NYC COVID-19 Testing Data](https://data.cityofnewyork.us/Health/COVID-19-Outcomes-by-Testing-Cohorts-Cases-Hospita/cwmx-mvra)
2. [NYC COVID-19 Hospitailzation Data](https://data.cityofnewyork.us/Health/COVID-19-Daily-Counts-of-Cases-Hospitalizations-an/rc75-m7u3)
4. [NYC COVID-19 Infection Rate Data](https://github.com/nychealth/coronavirus-data)
3. [CDC Vaccination Data](https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh)
### **Expected Outputs**: 
- Charts mapping out COVID-19 infection rates against other factors like vaccinations, age group, pre-existing conditions, season, etc.
- Choropleth visualizing infection rates and vaccination rates around the country using CDC data
Statistical analysis provided under every diagram thats created
### **Success Metric**: 
For this project to be considered successful, it would demonstrate and prove whether or not a causation exists between COVID-19 vaccinations and infection rates.  

### **Security and Privacy Considerations**: 
All data that is used in the project is under US public domain so there should be no issue with user data. Also, users are anonymous so there is no concern for HIPAA violations. 
 
