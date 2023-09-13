## Table of Contents

- [1. Introduction](#1.-Introduction)
- [2. Importing libraries](#2.-Importing-Libraries)
- [3. Loading dataset](#3.-Loading-Dataset)
- [4. Exploratory data analysis](#4.-Exploratory-Data-Analysis)
   - [4.1. Categorical variables](#4.1.-Analysis-of-Categorical-Variables)
   - [4.2. Numerical variables](#4.2.-Analysis-of-Numerical-Variables)
- [5. Data preprocessing](#5.-Data-Preprocessing)
   - [5.1. Label encoding](#5.1.-Label-Encoding)
   - [5.2. One hot encoding](#5.2.-One-Hot-Encoding)
   - [5.3. Correlation analysis](#5.3.-Correlation-Analysis)
   - [5.4. Outlier analysis](#5.4.-Outlier-Analysis)
   - [5.5. Missing data analysis](#5.5.-Missing-Data-Analysis)
   - [5.6. Feature scaling](#5.6.-Feature-Scaling)
- [6. Model development and evaluation](#6.-Model-Development-and-Evaluation)
   - [6.1. Logistic regression](#6.1.-Logistic-Regression)
   - [6.2. Decision Tree](#6.2.-Decision-Tree)
   - [6.3. Random Forest](#6.3.-Random-Forest)
   - [6.4. Gradient Boosting](#6.4.-Gradient-Boosting)
- [7. Feature importance](#7.-Feature-Importance)
- [8. Explaining the model](#8.-Explaining-the-Model)
- [9. Test prediction](#9.-Test-Prediction)
- [10. Conclusion](#10.-Conclusion)

### 1. Introduction

The dataset contains fictitious customer information of a telco company in California providing various service such as phone service, internet service, streaming service and more.

The data indicates if a user has churned, the services user signed up for and the users' demographic information.

\
**Identifier**
- `CustomerID`: A unique ID that identifies each customer.

**Independent variable**
- `Churn`: Yes = the customer left the company this quarter. No = the customer remained with the company.

**Demographic information**
- `Gender`: The customer’s gender: Male, Female
- `SeniorCitizen`: Indicates if the customer is 65 or older: Yes, No
- `Partner`: Indicates if the customer is married: Yes, No
- `Dependents`: Indicates if the customer lives with any dependents: Yes, No.

**Services**
- `PhoneService`: Indicates if the customer subscribes to home phone service with the company: Yes, No
- `MultipleLines`: Indicates if the customer subscribes to multiple telephone lines with the company: Yes, No
- `InternetService`: Indicates if the customer subscribes to Internet service with the company: No, DSL, Fiber Optic, Cable.
- `OnlineSecurity`: Indicates if the customer subscribes to an additional online security service provided by the company: Yes, No
- `OnlineBackup`: Indicates if the customer subscribes to an additional online backup service provided by the company: Yes, No
- `DeviceProtection`: Indicates if the customer subscribes to an additional device protection plan for their Internet equipment provided by the company: Yes, No
- `TechSupport`: Indicates if the customer subscribes to an additional technical support plan from the company with reduced wait times: Yes, No
- `StreamingTV`: Indicates if the customer uses their Internet service to stream television programing from a third party provider: Yes, No. The company does not charge an additional fee for this service.
- `StreamingMovies`: Indicates if the customer uses their Internet service to stream movies from a third party provider: Yes, No. The company does not charge an additional fee for this service.

**Customer account information**
- `tenure`: Indicates the total amount of months that the customer has been with the company by the end of the quarter specified above.
- `Contract`: Indicates the customer’s current contract type: Month-to-Month, One Year, Two Year.
- `PaperlessBilling`: Indicates if the customer has chosen paperless billing: Yes, No
- `PaymentMethod`: Indicates how the customer pays their bill: Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)
- `MonthlyCharges`: Indicates the customer’s current total monthly charge for all their services from the company.
- `TotalCharges`: The total amount charged to the customer

\
An analysis will be conducted to understand factors affecting customer churn. We also want to predict if a customer is going to churn.

### 2. Importing Libraries


```python
import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
%matplotlib inline
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
```

### 3. Loading Dataset


```python
df = pd.read_csv('Telco-Customer-Churn.csv')
```

### 4. Exploratory Data Analysis


```python
df.info()
df.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7043 entries, 0 to 7042
    Data columns (total 21 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   customerID        7043 non-null   object 
     1   gender            7043 non-null   object 
     2   SeniorCitizen     7043 non-null   int64  
     3   Partner           7043 non-null   object 
     4   Dependents        7043 non-null   object 
     5   tenure            7043 non-null   int64  
     6   PhoneService      7043 non-null   object 
     7   MultipleLines     7043 non-null   object 
     8   InternetService   7043 non-null   object 
     9   OnlineSecurity    7043 non-null   object 
     10  OnlineBackup      7043 non-null   object 
     11  DeviceProtection  7043 non-null   object 
     12  TechSupport       7043 non-null   object 
     13  StreamingTV       7043 non-null   object 
     14  StreamingMovies   7043 non-null   object 
     15  Contract          7043 non-null   object 
     16  PaperlessBilling  7043 non-null   object 
     17  PaymentMethod     7043 non-null   object 
     18  MonthlyCharges    7043 non-null   float64
     19  TotalCharges      7043 non-null   object 
     20  Churn             7043 non-null   object 
    dtypes: float64(1), int64(2), object(18)
    memory usage: 1.1+ MB





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customerID</th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>OnlineBackup</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7590-VHVEG</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>1</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5575-GNVDE</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>34</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>56.95</td>
      <td>1889.5</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3668-QPYBK</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7795-CFOCW</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>45</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1840.75</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9237-HQITU</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>



`>> From experience, errors in SQL joins can lead to duplicated rows.` \
`>> So let's check if there's any redundant records.`


```python
df.duplicated().sum()
```




    0



`>> No duplicated rows found, great!`


```python
df.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SeniorCitizen</th>
      <td>7043.0</td>
      <td>0.162147</td>
      <td>0.368612</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>tenure</th>
      <td>7043.0</td>
      <td>32.371149</td>
      <td>24.559481</td>
      <td>0.00</td>
      <td>9.0</td>
      <td>29.00</td>
      <td>55.00</td>
      <td>72.00</td>
    </tr>
    <tr>
      <th>MonthlyCharges</th>
      <td>7043.0</td>
      <td>64.761692</td>
      <td>30.090047</td>
      <td>18.25</td>
      <td>35.5</td>
      <td>70.35</td>
      <td>89.85</td>
      <td>118.75</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.nunique()
```




    customerID          7043
    gender                 2
    SeniorCitizen          2
    Partner                2
    Dependents             2
    tenure                73
    PhoneService           2
    MultipleLines          3
    InternetService        3
    OnlineSecurity         3
    OnlineBackup           3
    DeviceProtection       3
    TechSupport            3
    StreamingTV            3
    StreamingMovies        3
    Contract               3
    PaperlessBilling       2
    PaymentMethod          4
    MonthlyCharges      1585
    TotalCharges        6531
    Churn                  2
    dtype: int64



`>> High cardinality columns are non-categorical columns`

`>> Next, we group the features into two buckets (categorical and numerical) for easier analysis`


```python
categorical = df.drop(['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges'], axis=1)
numeric = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
```

`>> I'm curious to understand what are the categories available for each categorical feature`


```python
def categorical_list(category_col):
    df_values = pd.DataFrame()
    for col in category_col:
        values = category_col[col].unique().tolist()
        values_string = ', '.join(map(str, values))
        df_values[col] = [values_string]
    return df_values.rename({0:'data variation'}).T

categorical_list(categorical)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data variation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gender</th>
      <td>Female, Male</td>
    </tr>
    <tr>
      <th>SeniorCitizen</th>
      <td>0, 1</td>
    </tr>
    <tr>
      <th>Partner</th>
      <td>Yes, No</td>
    </tr>
    <tr>
      <th>Dependents</th>
      <td>No, Yes</td>
    </tr>
    <tr>
      <th>PhoneService</th>
      <td>No, Yes</td>
    </tr>
    <tr>
      <th>MultipleLines</th>
      <td>No phone service, No, Yes</td>
    </tr>
    <tr>
      <th>InternetService</th>
      <td>DSL, Fiber optic, No</td>
    </tr>
    <tr>
      <th>OnlineSecurity</th>
      <td>No, Yes, No internet service</td>
    </tr>
    <tr>
      <th>OnlineBackup</th>
      <td>Yes, No, No internet service</td>
    </tr>
    <tr>
      <th>DeviceProtection</th>
      <td>No, Yes, No internet service</td>
    </tr>
    <tr>
      <th>TechSupport</th>
      <td>No, Yes, No internet service</td>
    </tr>
    <tr>
      <th>StreamingTV</th>
      <td>No, Yes, No internet service</td>
    </tr>
    <tr>
      <th>StreamingMovies</th>
      <td>No, Yes, No internet service</td>
    </tr>
    <tr>
      <th>Contract</th>
      <td>Month-to-month, One year, Two year</td>
    </tr>
    <tr>
      <th>PaperlessBilling</th>
      <td>Yes, No</td>
    </tr>
    <tr>
      <th>PaymentMethod</th>
      <td>Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)</td>
    </tr>
    <tr>
      <th>Churn</th>
      <td>No, Yes</td>
    </tr>
  </tbody>
</table>
</div>



`>> Now, with a better understanding of the data, we need to make some adjustments.`


```python
# customerID is not needed
df.drop('customerID', axis=1, inplace=True)

# TotalCharges should be numeric
df['TotalCharges'] = df['TotalCharges'].apply(pd.to_numeric, errors='coerce')

# SeniorCitizen feature should be an object type, not int
df['SeniorCitizen'] = df['SeniorCitizen'].astype("O")
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7043 entries, 0 to 7042
    Data columns (total 20 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   gender            7043 non-null   object 
     1   SeniorCitizen     7043 non-null   object 
     2   Partner           7043 non-null   object 
     3   Dependents        7043 non-null   object 
     4   tenure            7043 non-null   int64  
     5   PhoneService      7043 non-null   object 
     6   MultipleLines     7043 non-null   object 
     7   InternetService   7043 non-null   object 
     8   OnlineSecurity    7043 non-null   object 
     9   OnlineBackup      7043 non-null   object 
     10  DeviceProtection  7043 non-null   object 
     11  TechSupport       7043 non-null   object 
     12  StreamingTV       7043 non-null   object 
     13  StreamingMovies   7043 non-null   object 
     14  Contract          7043 non-null   object 
     15  PaperlessBilling  7043 non-null   object 
     16  PaymentMethod     7043 non-null   object 
     17  MonthlyCharges    7043 non-null   float64
     18  TotalCharges      7032 non-null   float64
     19  Churn             7043 non-null   object 
    dtypes: float64(2), int64(1), object(17)
    memory usage: 1.1+ MB


#### 4.1. Analysis of Categorical Variables


```python
def cat_summary(dataframe, col):
    
    sns.set_style('whitegrid')
    
    for i in col:
        plt.figure(figsize=(8,4))
        print('\n' , 20*"—" , i, 20*"—")
        print(pd.DataFrame({"COUNT": col[i].value_counts(), "RATIO": round(100*col[i].value_counts()/len(col),2)}), end='\n\n\n')
        ax = sns.countplot(x=col[i], data=dataframe, hue='Churn', palette='OrRd')
        
        for container in ax.containers:
            ax.bar_label(container)
        
        plt.tight_layout()
        plt.show()



cat_summary(df, categorical)
```

    
     ———————————————————— gender ————————————————————
            COUNT  RATIO
    gender              
    Male     3555  50.48
    Female   3488  49.52
    
    



    
![png](output_22_1.png)
    


    
     ———————————————————— SeniorCitizen ————————————————————
                   COUNT  RATIO
    SeniorCitizen              
    0               5901  83.79
    1               1142  16.21
    
    



    
![png](output_22_3.png)
    


    
     ———————————————————— Partner ————————————————————
             COUNT  RATIO
    Partner              
    No        3641   51.7
    Yes       3402   48.3
    
    



    
![png](output_22_5.png)
    


    
     ———————————————————— Dependents ————————————————————
                COUNT  RATIO
    Dependents              
    No           4933  70.04
    Yes          2110  29.96
    
    



    
![png](output_22_7.png)
    


    
     ———————————————————— PhoneService ————————————————————
                  COUNT  RATIO
    PhoneService              
    Yes            6361  90.32
    No              682   9.68
    
    



    
![png](output_22_9.png)
    


    
     ———————————————————— MultipleLines ————————————————————
                      COUNT  RATIO
    MultipleLines                 
    No                 3390  48.13
    Yes                2971  42.18
    No phone service    682   9.68
    
    



    
![png](output_22_11.png)
    


    
     ———————————————————— InternetService ————————————————————
                     COUNT  RATIO
    InternetService              
    Fiber optic       3096  43.96
    DSL               2421  34.37
    No                1526  21.67
    
    



    
![png](output_22_13.png)
    


    
     ———————————————————— OnlineSecurity ————————————————————
                         COUNT  RATIO
    OnlineSecurity                   
    No                    3498  49.67
    Yes                   2019  28.67
    No internet service   1526  21.67
    
    



    
![png](output_22_15.png)
    


    
     ———————————————————— OnlineBackup ————————————————————
                         COUNT  RATIO
    OnlineBackup                     
    No                    3088  43.84
    Yes                   2429  34.49
    No internet service   1526  21.67
    
    



    
![png](output_22_17.png)
    


    
     ———————————————————— DeviceProtection ————————————————————
                         COUNT  RATIO
    DeviceProtection                 
    No                    3095  43.94
    Yes                   2422  34.39
    No internet service   1526  21.67
    
    



    
![png](output_22_19.png)
    


    
     ———————————————————— TechSupport ————————————————————
                         COUNT  RATIO
    TechSupport                      
    No                    3473  49.31
    Yes                   2044  29.02
    No internet service   1526  21.67
    
    



    
![png](output_22_21.png)
    


    
     ———————————————————— StreamingTV ————————————————————
                         COUNT  RATIO
    StreamingTV                      
    No                    2810  39.90
    Yes                   2707  38.44
    No internet service   1526  21.67
    
    



    
![png](output_22_23.png)
    


    
     ———————————————————— StreamingMovies ————————————————————
                         COUNT  RATIO
    StreamingMovies                  
    No                    2785  39.54
    Yes                   2732  38.79
    No internet service   1526  21.67
    
    



    
![png](output_22_25.png)
    


    
     ———————————————————— Contract ————————————————————
                    COUNT  RATIO
    Contract                    
    Month-to-month   3875  55.02
    Two year         1695  24.07
    One year         1473  20.91
    
    



    
![png](output_22_27.png)
    


    
     ———————————————————— PaperlessBilling ————————————————————
                      COUNT  RATIO
    PaperlessBilling              
    Yes                4171  59.22
    No                 2872  40.78
    
    



    
![png](output_22_29.png)
    


    
     ———————————————————— PaymentMethod ————————————————————
                               COUNT  RATIO
    PaymentMethod                          
    Electronic check            2365  33.58
    Mailed check                1612  22.89
    Bank transfer (automatic)   1544  21.92
    Credit card (automatic)     1522  21.61
    
    



    
![png](output_22_31.png)
    


    
     ———————————————————— Churn ————————————————————
           COUNT  RATIO
    Churn              
    No      5174  73.46
    Yes     1869  26.54
    
    



    
![png](output_22_33.png)
    


`>> Observations:`\
`    1. Gender appears to be well balanced when it comes to churn`\
`    2. While most users are not senior citizen, the churn rate for senior citizen is higher`\
`    3. Those who have a partner or dependents are less likely to churn`\
`    4. Customers on monthly contract has very high churn rate as compared to other contracts`\
`    5. Customers who make payment via electronic check has high churn rate as compared to other payment methods`\
`     6. Overall churn rate is at 26.5%
`

#### 4.2. Analysis of Numerical Variables


```python
print('\n' , 20*"—" , 'tenure', 20*"—")
ax = sns.histplot(data=df, x='tenure', hue='Churn', palette='pastel')

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()
plt.show()
```

    
     ———————————————————— tenure ————————————————————



    
![png](output_25_1.png)
    


`>> Churn is significantly higher when tenure is low`


```python
print('\n' , 20*"—" , 'MonthlyCharges', 20*"—")
ax = sns.histplot(data=df, x='MonthlyCharges', hue='Churn', palette='pastel')

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()
plt.show()
```

    
     ———————————————————— MonthlyCharges ————————————————————



    
![png](output_27_1.png)
    


`>> Churn rate seems to increase around monthly charges of 70-100`


```python
print('\n' , 20*"—" , 'TotalCharges', 20*"—")
ax = sns.histplot(data=df, x='TotalCharges', hue='Churn', palette='pastel')

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()
plt.show()
```

    
     ———————————————————— TotalCharges ————————————————————



    
![png](output_29_1.png)
    


`>> Churn rate is significantly high on the low end of total charges`

`>> Overall (tenure, monthly charges & total charges) seems to be telling the story that shorter tenure customers have high churn rate`

### 5. Data Preprocessing

#### 5.1. Label Encoding


```python
binary_cols = df.select_dtypes(exclude=np.number).loc[:, df.nunique() == 2]

def lbe(dataframe, col):
    labelencoder = LabelEncoder()
    
    for i in col:
        dataframe[i] = labelencoder.fit_transform(dataframe[i])
    return dataframe

lbe(df, binary_cols)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>OnlineBackup</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>1</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>34</td>
      <td>1</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>0</td>
      <td>Mailed check</td>
      <td>56.95</td>
      <td>1889.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>1</td>
      <td>Mailed check</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>0</td>
      <td>Bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1840.75</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>1</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7038</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>24</td>
      <td>1</td>
      <td>Yes</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>1</td>
      <td>Mailed check</td>
      <td>84.80</td>
      <td>1990.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7039</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>72</td>
      <td>1</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>1</td>
      <td>Credit card (automatic)</td>
      <td>103.20</td>
      <td>7362.90</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7040</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>11</td>
      <td>0</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>1</td>
      <td>Electronic check</td>
      <td>29.60</td>
      <td>346.45</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7041</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>1</td>
      <td>Mailed check</td>
      <td>74.40</td>
      <td>306.60</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7042</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>66</td>
      <td>1</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Two year</td>
      <td>1</td>
      <td>Bank transfer (automatic)</td>
      <td>105.65</td>
      <td>6844.50</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>7043 rows × 20 columns</p>
</div>




```python
ohe_cols = [col for col in categorical if (col not in binary_cols) and df[col].nunique() > 2]
ohe_cols
```




    ['MultipleLines',
     'InternetService',
     'OnlineSecurity',
     'OnlineBackup',
     'DeviceProtection',
     'TechSupport',
     'StreamingTV',
     'StreamingMovies',
     'Contract',
     'PaymentMethod']




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>OnlineBackup</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>1</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>34</td>
      <td>1</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>0</td>
      <td>Mailed check</td>
      <td>56.95</td>
      <td>1889.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>1</td>
      <td>Mailed check</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>0</td>
      <td>Bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1840.75</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>1</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Noticed how there's no phone service and no internet service category in some of the feature. It needs to be managed to avoid multicollinearity issue
df.replace({'No phone service': 'No', 'No internet service': 'No'}, inplace=True)
```

#### 5.2. One Hot Encoding


```python
ohe = OneHotEncoder(drop='if_binary') ## For features where it only has two values, we want to drop one column to prevent multicollinearity
ohe_values = ohe.fit_transform(df[ohe_cols]).toarray()
ohe_df = pd.DataFrame(ohe_values, columns=ohe.get_feature_names_out())

```


```python
df = pd.concat([df.drop(ohe_cols, axis=1),ohe_df], axis=1)
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7043 entries, 0 to 7042
    Data columns (total 27 columns):
     #   Column                                   Non-Null Count  Dtype  
    ---  ------                                   --------------  -----  
     0   gender                                   7043 non-null   int64  
     1   SeniorCitizen                            7043 non-null   int64  
     2   Partner                                  7043 non-null   int64  
     3   Dependents                               7043 non-null   int64  
     4   tenure                                   7043 non-null   int64  
     5   PhoneService                             7043 non-null   int64  
     6   PaperlessBilling                         7043 non-null   int64  
     7   MonthlyCharges                           7043 non-null   float64
     8   TotalCharges                             7032 non-null   float64
     9   Churn                                    7043 non-null   int64  
     10  MultipleLines_Yes                        7043 non-null   float64
     11  InternetService_DSL                      7043 non-null   float64
     12  InternetService_Fiber optic              7043 non-null   float64
     13  InternetService_No                       7043 non-null   float64
     14  OnlineSecurity_Yes                       7043 non-null   float64
     15  OnlineBackup_Yes                         7043 non-null   float64
     16  DeviceProtection_Yes                     7043 non-null   float64
     17  TechSupport_Yes                          7043 non-null   float64
     18  StreamingTV_Yes                          7043 non-null   float64
     19  StreamingMovies_Yes                      7043 non-null   float64
     20  Contract_Month-to-month                  7043 non-null   float64
     21  Contract_One year                        7043 non-null   float64
     22  Contract_Two year                        7043 non-null   float64
     23  PaymentMethod_Bank transfer (automatic)  7043 non-null   float64
     24  PaymentMethod_Credit card (automatic)    7043 non-null   float64
     25  PaymentMethod_Electronic check           7043 non-null   float64
     26  PaymentMethod_Mailed check               7043 non-null   float64
    dtypes: float64(19), int64(8)
    memory usage: 1.5 MB


#### 5.3. Correlation Analysis


```python

fig, ax = plt.subplots(figsize=(20, 20))

corr = df.corr()
matrix = np.triu(corr)

# color map
cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)

# plot heatmap
sns.set_style('white')
heatmap = sns.heatmap(df.corr(), annot=True, mask=matrix, fmt=".2f", cmap='RdBu', vmin=-1, vmax=1, cbar_kws={"shrink": .8})

plt.show()
```


    
![png](output_41_0.png)
    



```python
positive_corr = round(df.corr()['Churn'].sort_values(ascending=False)[lambda x: x > 0][lambda x: x < 1],2)
negative_corr = round(df.corr()['Churn'].sort_values(ascending=True)[lambda x: x > -1][lambda x: x < 0],2)
```


```python
ax = sns.barplot(x=positive_corr.values, y=positive_corr.index, palette = 'Reds_r')


ax.bar_label(container, label_type='edge', padding=3)

ax.set_xticklabels([])
sns.despine(left=True, bottom=True)
```


    
![png](output_43_0.png)
    


`>> Contract month to month has 0.41 correlation to churn.`\
`>> Internet service fiber optic has 0.31 correlation to churn.`\
`>> Electronic check payment method has 0.3 correlation to churn.`


```python
ax = sns.barplot(x=negative_corr.values, y=negative_corr.index, palette = 'Blues_r')

for container in ax.containers:
    ax.bar_label(container, label_type='edge', padding=3)
    
# ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.set_xticklabels([])

sns.despine(left=True, bottom=True)

plt.show()
```


    
![png](output_45_0.png)
    


`>> Tenure has -0.35 correlation to churn.`\
`>> Two year contract has -0.3 correlation to churn.`\
`>> No internet service has -0.23 correlation to churn.`

`>> In essence, longer tenure has negative correlation to churn`

#### 5.4. Outlier Analysis

`>> Check for outliers using the IQR method`


```python
def outlier_limits(dataframe, col, q1=0.25, q3=0.75):
    quartile1 = dataframe[col].quantile(q1)
    quartile3 = dataframe[col].quantile(q3)
    IQR = quartile3 - quartile1
    up_limit = quartile3 + IQR*1.5
    low_limit = quartile1 - IQR*1.5
    return up_limit, low_limit
```


```python
def check_outlier(dataframe, col):
    up_limit, low_limit = outlier_limits(dataframe, col)
    
    if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
        return True
    else:
        return False
```


```python
for col in numeric:
    print(col, check_outlier(df, col))
```

    tenure False
    MonthlyCharges False
    TotalCharges False


`>> No outlier found`

#### 5.5. Missing Data Analysis


```python
na_columns = [col for col in df.columns if df[col].isnull().sum() > 0]

missing_df = []

for col in na_columns:
    missing_rows = df[col].isnull().sum()
    percent_missing = round(df[col].isnull().sum()/df[col].shape[0]*100,2)

    result = pd.DataFrame({
                        "Total missing rows": missing_rows, 
                        "Percent of missing values": percent_missing 
                        }, index=[col])
    missing_df.append(result)

final_result = pd.concat(missing_df)
final_result
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total missing rows</th>
      <th>Percent of missing values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TotalCharges</th>
      <td>11</td>
      <td>0.16</td>
    </tr>
  </tbody>
</table>
</div>



`>> Let's dig deeper to understand if it makes sense for TotalCharges to have missing values`


```python
df[df['TotalCharges'].isnull()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>PaperlessBilling</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
      <th>MultipleLines_Yes</th>
      <th>InternetService_DSL</th>
      <th>InternetService_Fiber optic</th>
      <th>InternetService_No</th>
      <th>OnlineSecurity_Yes</th>
      <th>OnlineBackup_Yes</th>
      <th>DeviceProtection_Yes</th>
      <th>TechSupport_Yes</th>
      <th>StreamingTV_Yes</th>
      <th>StreamingMovies_Yes</th>
      <th>Contract_Month-to-month</th>
      <th>Contract_One year</th>
      <th>Contract_Two year</th>
      <th>PaymentMethod_Bank transfer (automatic)</th>
      <th>PaymentMethod_Credit card (automatic)</th>
      <th>PaymentMethod_Electronic check</th>
      <th>PaymentMethod_Mailed check</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>488</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>52.55</td>
      <td>NaN</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>753</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>20.25</td>
      <td>NaN</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>936</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>80.85</td>
      <td>NaN</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1082</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>25.75</td>
      <td>NaN</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1340</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>56.05</td>
      <td>NaN</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3331</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>19.85</td>
      <td>NaN</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3826</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>25.35</td>
      <td>NaN</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4380</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>20.00</td>
      <td>NaN</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5218</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>19.70</td>
      <td>NaN</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6670</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>73.35</td>
      <td>NaN</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6754</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>61.90</td>
      <td>NaN</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



`>> Few things I noticed here:`\
`    1. All users with blank TotalCharges has 0 as tenure`\
`    2. All users with blank TotalCharges has not churn`\
`    3. While TotalCharges are blank, MonthlyCharges are not.`

`>> I believe this means that these customers are still an active customer who has not completed their first month with the company`

`>> Would it makes sense to just set MonthlyCharges as the TotalCharges?`


```python
check_totalcharges = df.copy()
check_totalcharges['tc_mc_multiple'] = (df['TotalCharges']/df['MonthlyCharges'])
check_totalcharges[['tenure', 'TotalCharges', 'MonthlyCharges', 'tc_mc_multiple']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tenure</th>
      <th>TotalCharges</th>
      <th>MonthlyCharges</th>
      <th>tc_mc_multiple</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34</td>
      <td>1889.50</td>
      <td>56.95</td>
      <td>33.178227</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>108.15</td>
      <td>53.85</td>
      <td>2.008357</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>1840.75</td>
      <td>42.30</td>
      <td>43.516548</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>151.65</td>
      <td>70.70</td>
      <td>2.144979</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7038</th>
      <td>24</td>
      <td>1990.50</td>
      <td>84.80</td>
      <td>23.472877</td>
    </tr>
    <tr>
      <th>7039</th>
      <td>72</td>
      <td>7362.90</td>
      <td>103.20</td>
      <td>71.345930</td>
    </tr>
    <tr>
      <th>7040</th>
      <td>11</td>
      <td>346.45</td>
      <td>29.60</td>
      <td>11.704392</td>
    </tr>
    <tr>
      <th>7041</th>
      <td>4</td>
      <td>306.60</td>
      <td>74.40</td>
      <td>4.120968</td>
    </tr>
    <tr>
      <th>7042</th>
      <td>66</td>
      <td>6844.50</td>
      <td>105.65</td>
      <td>64.784666</td>
    </tr>
  </tbody>
</table>
<p>7043 rows × 4 columns</p>
</div>



`>> With the assumption that 'TotalCharges' represents the cumulative charges of a customer and that it is a result of total tenure multiplied by monthly charges.`

`>> I created a column to check the above assumption, named as 'tc_mc_multiple'.`

`>> tc_mc_multiple column should result in whole number values if my assumption holds.`

`>> Here we can observe that:`\
`   1. When tenure is 1 month, the TotalCharges would be equivalent to the MonthlyCharges.`\
`   2. When tenure is 2 months or more, the TotalCharges does not necessarily equates to MonthlyCharges. I think that makes sense, if there's any additional fee imposed on customers like late payment fee or additional ad hoc services, it wouldn't be part of monthly charges.`

`It's safe to just fill the missing values with the TotalCharges value.`


```python
df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)
```


```python
df['TotalCharges'].isnull().sum()
```




    0



#### 5.6. Feature Scaling


```python
scaler = StandardScaler()
```


```python
df[numeric.columns.tolist()] = scaler.fit_transform(df[numeric.columns.tolist()])
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>PaperlessBilling</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
      <th>MultipleLines_Yes</th>
      <th>InternetService_DSL</th>
      <th>InternetService_Fiber optic</th>
      <th>InternetService_No</th>
      <th>OnlineSecurity_Yes</th>
      <th>OnlineBackup_Yes</th>
      <th>DeviceProtection_Yes</th>
      <th>TechSupport_Yes</th>
      <th>StreamingTV_Yes</th>
      <th>StreamingMovies_Yes</th>
      <th>Contract_Month-to-month</th>
      <th>Contract_One year</th>
      <th>Contract_Two year</th>
      <th>PaymentMethod_Bank transfer (automatic)</th>
      <th>PaymentMethod_Credit card (automatic)</th>
      <th>PaymentMethod_Electronic check</th>
      <th>PaymentMethod_Mailed check</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-1.277445</td>
      <td>0</td>
      <td>1</td>
      <td>-1.160323</td>
      <td>-0.992667</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.066327</td>
      <td>1</td>
      <td>0</td>
      <td>-0.259629</td>
      <td>-0.172198</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1.236724</td>
      <td>1</td>
      <td>1</td>
      <td>-0.362660</td>
      <td>-0.958122</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.514251</td>
      <td>0</td>
      <td>0</td>
      <td>-0.746535</td>
      <td>-0.193706</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1.236724</td>
      <td>1</td>
      <td>1</td>
      <td>0.197365</td>
      <td>-0.938930</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## 6. Model Development and Evaluation


```python
X = df.drop('Churn', axis=1)
y = df['Churn']
```


```python
X_train, X_test, y_train, y_test = train_test_split(
...     X, y, test_size=0.33, random_state=42)
```


```python
# Using stratified KFold as the dataset is not balanced
skfold = StratifiedKFold(n_splits=5)
```


```python
def check_scoring(scorings):

    for x in scorings:
        if x not in ('fit_time', 'score_time'):
            print(f"{x[5:].capitalize()}: {round(scorings[x].mean(),3)}")
```

### 6.1. Logistic Regression


```python
lm = LogisticRegression()
```


```python
lm_scores = cross_validate(lm, X, y, cv=skfold, scoring=["accuracy","precision", "recall", "f1", "roc_auc"])
```


```python
check_scoring(lm_scores)
```

    Accuracy: 0.804
    Precision: 0.655
    Recall: 0.553
    F1: 0.6
    Roc_auc: 0.845


### 6.2. Decision Tree


```python
decision_tree = DecisionTreeClassifier()
decision_tree_scores = cross_validate(decision_tree, X, y, cv=skfold, scoring=["accuracy","precision", "recall", "f1", "roc_auc"])
```


```python
check_scoring(decision_tree_scores)
```

    Accuracy: 0.728
    Precision: 0.488
    Recall: 0.491
    F1: 0.489
    Roc_auc: 0.653


### 6.3. Random Forest


```python
import shap
rfc = RandomForestClassifier()
rfc_scores = cross_validate(rfc, X, y, cv=skfold, scoring=["accuracy","precision", "recall", "f1", "roc_auc"])
check_scoring(rfc_scores)
```

    Accuracy: 0.787
    Precision: 0.631
    Recall: 0.479
    F1: 0.545
    Roc_auc: 0.824



```python
rf_params = {
    'n_estimators': [100, 200, 300], 
    'max_depth': [None,2,3,5,10,20], 
    'max_features': ['sqrt',2,4,8,16,'log2', None]}

rf_best_grid = GridSearchCV(rfc, rf_params, cv=skfold, n_jobs=-1, verbose=2).fit(X_train, y_train)

rf_best_grid.best_params_

rf_final = rfc.set_params(**rf_best_grid.best_params_).fit(X_train, y_train)
```

    Fitting 5 folds for each of 126 candidates, totalling 630 fits



```python
rfc_scores = cross_validate(rf_final, X, y, cv=skfold, scoring=["accuracy","precision", "recall", "f1", "roc_auc"])
check_scoring(rfc_scores)
```

    Accuracy: 0.801
    Precision: 0.662
    Recall: 0.514
    F1: 0.579
    Roc_auc: 0.844


### 6.4. Gradient Boosting


```python
gbm = GradientBoostingClassifier()
gbm_scores = cross_validate(gbm, X, y, cv=skfold, scoring=["accuracy","precision", "recall", "f1", "roc_auc"])
check_scoring(gbm_scores)
```

    Accuracy: 0.804
    Precision: 0.666
    Recall: 0.525
    F1: 0.587
    Roc_auc: 0.846



```python
gbm_params = {
    'learning_rate': [0.01, 0.1], 
    'max_depth': [2,3,5,10,20], 
    'n_estimators': [100, 200, 500],
    'subsample': [0.5, 0.7, 0.9, 1]}

gbm_best_grid = GridSearchCV(gbm, gbm_params, cv=skfold, n_jobs=-1, verbose=2).fit(X_train, y_train)

gbm_best_grid.best_params_

gbm_final = gbm.set_params(**gbm_best_grid.best_params_).fit(X_train, y_train)
```

    Fitting 5 folds for each of 120 candidates, totalling 600 fits
    [CV] END max_depth=None, max_features=sqrt, n_estimators=100; total time=   0.5s
    [CV] END max_depth=None, max_features=sqrt, n_estimators=200; total time=   1.2s
    [CV] END ...max_depth=None, max_features=2, n_estimators=100; total time=   0.4s
    [CV] END ...max_depth=None, max_features=2, n_estimators=200; total time=   0.7s
    [CV] END ...max_depth=None, max_features=2, n_estimators=300; total time=   1.1s
    [CV] END ...max_depth=None, max_features=4, n_estimators=300; total time=   1.4s
    [CV] END ...max_depth=None, max_features=8, n_estimators=200; total time=   1.2s
    [CV] END ...max_depth=None, max_features=8, n_estimators=300; total time=   2.0s
    [CV] END ..max_depth=None, max_features=16, n_estimators=300; total time=   2.7s
    [CV] END max_depth=None, max_features=log2, n_estimators=200; total time=   0.9s
    [CV] END max_depth=None, max_features=log2, n_estimators=300; total time=   1.4s
    [CV] END max_depth=None, max_features=None, n_estimators=200; total time=   2.8s
    [CV] END ...max_depth=2, max_features=sqrt, n_estimators=100; total time=   0.2s
    [CV] END ...max_depth=2, max_features=sqrt, n_estimators=100; total time=   0.2s
    [CV] END ...max_depth=2, max_features=sqrt, n_estimators=100; total time=   0.2s
    [CV] END ...max_depth=2, max_features=sqrt, n_estimators=200; total time=   0.4s
    [CV] END ...max_depth=2, max_features=sqrt, n_estimators=200; total time=   0.4s
    [CV] END ...max_depth=2, max_features=sqrt, n_estimators=300; total time=   0.6s
    [CV] END ......max_depth=2, max_features=2, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=2, max_features=2, n_estimators=200; total time=   0.3s
    [CV] END ......max_depth=2, max_features=2, n_estimators=300; total time=   0.5s
    [CV] END ......max_depth=2, max_features=4, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=2, max_features=4, n_estimators=300; total time=   0.6s
    [CV] END ......max_depth=2, max_features=8, n_estimators=200; total time=   0.5s
    [CV] END ......max_depth=2, max_features=8, n_estimators=300; total time=   0.7s
    [CV] END .....max_depth=2, max_features=16, n_estimators=300; total time=   0.9s
    [CV] END ...max_depth=2, max_features=log2, n_estimators=200; total time=   0.4s
    [CV] END ...max_depth=2, max_features=log2, n_estimators=300; total time=   0.6s
    [CV] END ...max_depth=2, max_features=None, n_estimators=200; total time=   0.8s
    [CV] END ...max_depth=3, max_features=sqrt, n_estimators=100; total time=   0.2s
    [CV] END ...max_depth=3, max_features=sqrt, n_estimators=100; total time=   0.2s
    [CV] END ...max_depth=3, max_features=sqrt, n_estimators=200; total time=   0.4s
    [CV] END ...max_depth=3, max_features=sqrt, n_estimators=300; total time=   0.7s
    [CV] END ......max_depth=3, max_features=2, n_estimators=300; total time=   0.6s
    [CV] END ......max_depth=3, max_features=4, n_estimators=200; total time=   0.5s
    [CV] END ......max_depth=3, max_features=8, n_estimators=100; total time=   0.3s
    [CV] END ......max_depth=3, max_features=8, n_estimators=200; total time=   0.5s
    [CV] END ......max_depth=3, max_features=8, n_estimators=300; total time=   0.9s
    [CV] END .....max_depth=3, max_features=16, n_estimators=300; total time=   1.3s
    [CV] END ...max_depth=3, max_features=log2, n_estimators=200; total time=   0.5s
    [CV] END ...max_depth=3, max_features=None, n_estimators=100; total time=   0.5s
    [CV] END ...max_depth=3, max_features=None, n_estimators=200; total time=   1.0s
    [CV] END ...max_depth=3, max_features=None, n_estimators=300; total time=   1.5s
    [CV] END ......max_depth=5, max_features=2, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=5, max_features=2, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=5, max_features=2, n_estimators=200; total time=   0.4s
    [CV] END ......max_depth=5, max_features=4, n_estimators=100; total time=   0.3s
    [CV] END ......max_depth=5, max_features=4, n_estimators=200; total time=   0.5s
    [CV] END ......max_depth=5, max_features=4, n_estimators=300; total time=   0.8s
    [CV] END ......max_depth=5, max_features=8, n_estimators=300; total time=   1.0s
    [CV] END .....max_depth=5, max_features=16, n_estimators=200; total time=   1.0s
    [CV] END .....max_depth=5, max_features=16, n_estimators=300; total time=   1.5s
    [CV] END ...max_depth=5, max_features=None, n_estimators=100; total time=   0.7s
    [CV] END ...max_depth=5, max_features=None, n_estimators=200; total time=   1.5s
    [CV] END ..max_depth=10, max_features=sqrt, n_estimators=100; total time=   0.4s
    [CV] END ..max_depth=10, max_features=sqrt, n_estimators=200; total time=   0.8s
    [CV] END ..max_depth=10, max_features=sqrt, n_estimators=300; total time=   1.2s
    [CV] END .....max_depth=10, max_features=2, n_estimators=200; total time=   0.6s
    [CV] END .....max_depth=10, max_features=4, n_estimators=100; total time=   0.4s
    [CV] END .....max_depth=10, max_features=4, n_estimators=200; total time=   0.8s
    [CV] END .....max_depth=10, max_features=4, n_estimators=300; total time=   1.1s
    [CV] END .....max_depth=10, max_features=8, n_estimators=200; total time=   1.0s
    [CV] END ....max_depth=10, max_features=16, n_estimators=100; total time=   0.8s
    [CV] END ....max_depth=10, max_features=16, n_estimators=200; total time=   1.6s
    [CV] END ..max_depth=10, max_features=log2, n_estimators=100; total time=   0.4s
    [CV] END ..max_depth=10, max_features=log2, n_estimators=100; total time=   0.4s
    [CV] END ..max_depth=10, max_features=log2, n_estimators=200; total time=   0.8s
    [CV] END ..max_depth=10, max_features=log2, n_estimators=300; total time=   1.2s
    [CV] END ..max_depth=10, max_features=None, n_estimators=200; total time=   2.4s
    [CV] END ..max_depth=10, max_features=None, n_estimators=300; total time=   3.6s
    [CV] END .....max_depth=20, max_features=2, n_estimators=200; total time=   0.8s
    [CV] END .....max_depth=20, max_features=2, n_estimators=300; total time=   1.3s
    [CV] END .....max_depth=20, max_features=4, n_estimators=200; total time=   1.2s
    [CV] END .....max_depth=20, max_features=8, n_estimators=100; total time=   0.7s
    [CV] END .....max_depth=20, max_features=8, n_estimators=200; total time=   1.2s
    [CV] END ....max_depth=20, max_features=16, n_estimators=100; total time=   0.9s
    [CV] END ....max_depth=20, max_features=16, n_estimators=100; total time=   1.0s
    [CV] END ....max_depth=20, max_features=16, n_estimators=300; total time=   3.1s
    [CV] END ..max_depth=20, max_features=log2, n_estimators=300; total time=   1.6s
    [CV] END ..max_depth=20, max_features=None, n_estimators=100; total time=   2.0s
    [CV] END ..max_depth=20, max_features=None, n_estimators=300; total time=   3.7s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=100, subsample=0.7; total time=   0.4s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=100, subsample=0.9; total time=   0.6s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=200, subsample=0.5; total time=   0.7s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=200, subsample=0.9; total time=   0.9s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=200, subsample=1; total time=   0.9s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=500, subsample=0.7; total time=   1.8s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=500, subsample=0.9; total time=   2.3s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.5; total time=   0.5s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.7; total time=   0.5s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.9; total time=   0.6s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=200, subsample=0.5; total time=   0.9s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=200, subsample=0.7; total time=   1.1s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=200, subsample=1; total time=   1.3s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=500, subsample=0.5; total time=   2.3s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=500, subsample=0.9; total time=   3.8s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=100, subsample=0.5; total time=   0.8s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=100, subsample=0.5; total time=   0.8s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=100, subsample=0.7; total time=   1.4s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=100, subsample=0.7; total time=   1.1s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=100, subsample=1; total time=   1.6s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=200, subsample=0.5; total time=   1.7s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=200, subsample=0.9; total time=   2.3s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=200, subsample=1; total time=   2.1s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=500, subsample=0.7; total time=   4.2s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=500, subsample=1; total time=   7.0s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=100, subsample=0.5; total time=   2.4s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=100, subsample=0.7; total time=   3.2s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=100, subsample=1; total time=   3.2s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=200, subsample=0.7; total time=   6.5s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=200, subsample=0.9; total time=   7.6s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=500, subsample=0.5; total time=  11.0s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=500, subsample=0.7; total time=  12.6s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=500, subsample=1; total time=  15.6s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=100, subsample=0.7; total time=   9.1s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=100, subsample=0.9; total time=   9.8s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=200, subsample=0.7; total time=  18.4s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=200, subsample=0.9; total time=  21.7s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=500, subsample=0.7; total time=  44.9s


    A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.


    [CV] END max_depth=None, max_features=sqrt, n_estimators=200; total time=   1.2s
    [CV] END ...max_depth=None, max_features=2, n_estimators=100; total time=   0.5s
    [CV] END ...max_depth=None, max_features=2, n_estimators=100; total time=   0.4s
    [CV] END ...max_depth=None, max_features=2, n_estimators=200; total time=   0.8s
    [CV] END ...max_depth=None, max_features=2, n_estimators=300; total time=   1.2s
    [CV] END ...max_depth=None, max_features=4, n_estimators=300; total time=   1.4s
    [CV] END ...max_depth=None, max_features=8, n_estimators=200; total time=   1.2s
    [CV] END ...max_depth=None, max_features=8, n_estimators=300; total time=   1.8s
    [CV] END ..max_depth=None, max_features=16, n_estimators=200; total time=   1.9s
    [CV] END max_depth=None, max_features=log2, n_estimators=100; total time=   0.5s
    [CV] END max_depth=None, max_features=log2, n_estimators=200; total time=   1.0s
    [CV] END max_depth=None, max_features=log2, n_estimators=300; total time=   1.4s
    [CV] END max_depth=None, max_features=None, n_estimators=200; total time=   2.9s
    [CV] END max_depth=None, max_features=None, n_estimators=300; total time=   4.0s
    [CV] END ......max_depth=2, max_features=8, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=2, max_features=8, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=2, max_features=8, n_estimators=200; total time=   0.5s
    [CV] END .....max_depth=2, max_features=16, n_estimators=100; total time=   0.3s
    [CV] END .....max_depth=2, max_features=16, n_estimators=200; total time=   0.6s
    [CV] END .....max_depth=2, max_features=16, n_estimators=300; total time=   0.9s
    [CV] END ...max_depth=2, max_features=None, n_estimators=100; total time=   0.4s
    [CV] END ...max_depth=2, max_features=None, n_estimators=100; total time=   0.4s
    [CV] END ...max_depth=2, max_features=None, n_estimators=300; total time=   1.2s
    [CV] END ...max_depth=3, max_features=sqrt, n_estimators=300; total time=   0.6s
    [CV] END ......max_depth=3, max_features=2, n_estimators=200; total time=   0.4s
    [CV] END ......max_depth=3, max_features=2, n_estimators=300; total time=   0.5s
    [CV] END ......max_depth=3, max_features=4, n_estimators=300; total time=   0.7s
    [CV] END ......max_depth=3, max_features=8, n_estimators=100; total time=   0.3s
    [CV] END ......max_depth=3, max_features=8, n_estimators=300; total time=   0.8s
    [CV] END .....max_depth=3, max_features=16, n_estimators=100; total time=   0.4s
    [CV] END .....max_depth=3, max_features=16, n_estimators=300; total time=   1.5s
    [CV] END ...max_depth=3, max_features=log2, n_estimators=300; total time=   0.7s
    [CV] END ...max_depth=3, max_features=None, n_estimators=100; total time=   0.5s
    [CV] END ...max_depth=3, max_features=None, n_estimators=300; total time=   1.6s
    [CV] END ...max_depth=5, max_features=sqrt, n_estimators=300; total time=   0.9s
    [CV] END ......max_depth=5, max_features=2, n_estimators=200; total time=   0.4s
    [CV] END ......max_depth=5, max_features=4, n_estimators=100; total time=   0.3s
    [CV] END ......max_depth=5, max_features=4, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=5, max_features=4, n_estimators=200; total time=   0.5s
    [CV] END ......max_depth=5, max_features=8, n_estimators=100; total time=   0.3s
    [CV] END ......max_depth=5, max_features=8, n_estimators=200; total time=   0.7s
    [CV] END .....max_depth=5, max_features=16, n_estimators=100; total time=   0.5s
    [CV] END .....max_depth=5, max_features=16, n_estimators=100; total time=   0.5s
    [CV] END .....max_depth=5, max_features=16, n_estimators=300; total time=   1.5s
    [CV] END ...max_depth=5, max_features=log2, n_estimators=300; total time=   0.8s
    [CV] END ...max_depth=5, max_features=None, n_estimators=100; total time=   0.7s
    [CV] END ...max_depth=5, max_features=None, n_estimators=300; total time=   2.1s
    [CV] END ..max_depth=10, max_features=sqrt, n_estimators=200; total time=   0.8s
    [CV] END .....max_depth=10, max_features=2, n_estimators=100; total time=   0.3s
    [CV] END .....max_depth=10, max_features=2, n_estimators=100; total time=   0.3s
    [CV] END .....max_depth=10, max_features=2, n_estimators=200; total time=   0.6s
    [CV] END .....max_depth=10, max_features=4, n_estimators=100; total time=   0.4s
    [CV] END .....max_depth=10, max_features=4, n_estimators=100; total time=   0.4s
    [CV] END .....max_depth=10, max_features=4, n_estimators=300; total time=   1.2s
    [CV] END .....max_depth=10, max_features=8, n_estimators=100; total time=   0.5s
    [CV] END .....max_depth=10, max_features=8, n_estimators=300; total time=   1.6s
    [CV] END ....max_depth=10, max_features=16, n_estimators=200; total time=   1.6s
    [CV] END ....max_depth=10, max_features=16, n_estimators=300; total time=   2.5s
    [CV] END ..max_depth=10, max_features=None, n_estimators=100; total time=   1.2s
    [CV] END ..max_depth=10, max_features=None, n_estimators=200; total time=   2.3s
    [CV] END ..max_depth=20, max_features=sqrt, n_estimators=100; total time=   0.5s
    [CV] END ..max_depth=20, max_features=sqrt, n_estimators=200; total time=   1.0s
    [CV] END ..max_depth=20, max_features=sqrt, n_estimators=300; total time=   1.5s
    [CV] END .....max_depth=20, max_features=2, n_estimators=100; total time=   0.4s
    [CV] END .....max_depth=20, max_features=2, n_estimators=200; total time=   0.8s
    [CV] END .....max_depth=20, max_features=4, n_estimators=100; total time=   0.5s
    [CV] END .....max_depth=20, max_features=4, n_estimators=200; total time=   1.3s
    [CV] END .....max_depth=20, max_features=8, n_estimators=100; total time=   0.7s
    [CV] END .....max_depth=20, max_features=8, n_estimators=100; total time=   0.7s
    [CV] END .....max_depth=20, max_features=8, n_estimators=200; total time=   1.3s
    [CV] END ....max_depth=20, max_features=16, n_estimators=100; total time=   1.0s
    [CV] END ....max_depth=20, max_features=16, n_estimators=200; total time=   2.0s
    [CV] END ..max_depth=20, max_features=log2, n_estimators=100; total time=   0.7s
    [CV] END ..max_depth=20, max_features=log2, n_estimators=100; total time=   0.5s
    [CV] END ..max_depth=20, max_features=log2, n_estimators=200; total time=   1.0s
    [CV] END ..max_depth=20, max_features=log2, n_estimators=300; total time=   1.9s
    [CV] END ..max_depth=20, max_features=None, n_estimators=200; total time=   2.9s
    [CV] END ..max_depth=20, max_features=None, n_estimators=300; total time=   3.2s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=100, subsample=0.5; total time=   0.3s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=100, subsample=0.9; total time=   0.5s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=200, subsample=0.5; total time=   0.6s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=200, subsample=0.7; total time=   0.7s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=200, subsample=1; total time=   0.9s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=500, subsample=0.5; total time=   1.6s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=500, subsample=0.9; total time=   2.3s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.5; total time=   0.5s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.5; total time=   0.5s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.7; total time=   0.5s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.9; total time=   0.6s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=100, subsample=1; total time=   0.6s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=200, subsample=0.7; total time=   1.1s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=200, subsample=0.9; total time=   1.3s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=500, subsample=0.5; total time=   2.3s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=500, subsample=0.7; total time=   3.1s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=500, subsample=1; total time=   4.3s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=100, subsample=0.9; total time=   1.4s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=200, subsample=0.5; total time=   1.9s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=200, subsample=0.7; total time=   2.1s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=200, subsample=1; total time=   2.2s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=500, subsample=0.5; total time=   3.4s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=500, subsample=0.9; total time=   6.7s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=100, subsample=0.5; total time=   2.5s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=100, subsample=0.5; total time=   2.5s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=100, subsample=0.9; total time=   3.7s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=200, subsample=0.5; total time=   4.9s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=200, subsample=0.7; total time=   7.1s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=200, subsample=1; total time=   7.1s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=500, subsample=0.5; total time=  10.9s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=500, subsample=0.9; total time=  15.2s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=100, subsample=0.5; total time=   7.3s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=100, subsample=0.5; total time=   7.2s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=100, subsample=0.9; total time=   9.9s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=200, subsample=0.5; total time=  14.3s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=200, subsample=0.7; total time=  20.1s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=200, subsample=1; total time=   9.1s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=500, subsample=0.7; total time=  46.8s
    [CV] END max_depth=None, max_features=sqrt, n_estimators=200; total time=   1.1s
    [CV] END max_depth=None, max_features=sqrt, n_estimators=300; total time=   1.6s
    [CV] END ...max_depth=None, max_features=2, n_estimators=300; total time=   1.2s
    [CV] END ...max_depth=None, max_features=4, n_estimators=200; total time=   1.0s
    [CV] END ...max_depth=None, max_features=8, n_estimators=100; total time=   0.6s
    [CV] END ...max_depth=None, max_features=8, n_estimators=200; total time=   1.2s
    [CV] END ..max_depth=None, max_features=16, n_estimators=100; total time=   1.0s
    [CV] END ..max_depth=None, max_features=16, n_estimators=100; total time=   1.0s
    [CV] END ..max_depth=None, max_features=16, n_estimators=300; total time=   2.8s
    [CV] END max_depth=None, max_features=log2, n_estimators=200; total time=   1.0s
    [CV] END max_depth=None, max_features=None, n_estimators=100; total time=   1.6s
    [CV] END max_depth=None, max_features=None, n_estimators=200; total time=   2.9s
    [CV] END ...max_depth=2, max_features=sqrt, n_estimators=100; total time=   0.2s
    [CV] END ...max_depth=2, max_features=sqrt, n_estimators=200; total time=   0.4s
    [CV] END ...max_depth=2, max_features=sqrt, n_estimators=300; total time=   0.7s
    [CV] END ...max_depth=2, max_features=sqrt, n_estimators=300; total time=   0.7s
    [CV] END ......max_depth=2, max_features=2, n_estimators=200; total time=   0.4s
    [CV] END ......max_depth=2, max_features=4, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=2, max_features=4, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=2, max_features=4, n_estimators=200; total time=   0.4s
    [CV] END ......max_depth=2, max_features=4, n_estimators=300; total time=   0.6s
    [CV] END ......max_depth=2, max_features=8, n_estimators=300; total time=   0.7s
    [CV] END .....max_depth=2, max_features=16, n_estimators=100; total time=   0.3s
    [CV] END .....max_depth=2, max_features=16, n_estimators=300; total time=   0.9s
    [CV] END ...max_depth=2, max_features=log2, n_estimators=300; total time=   0.6s
    [CV] END ...max_depth=2, max_features=None, n_estimators=100; total time=   0.4s
    [CV] END ...max_depth=2, max_features=None, n_estimators=300; total time=   1.2s
    [CV] END ...max_depth=3, max_features=sqrt, n_estimators=200; total time=   0.4s
    [CV] END ......max_depth=3, max_features=2, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=3, max_features=2, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=3, max_features=2, n_estimators=200; total time=   0.4s
    [CV] END ......max_depth=3, max_features=4, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=3, max_features=4, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=3, max_features=4, n_estimators=300; total time=   0.7s
    [CV] END ......max_depth=3, max_features=8, n_estimators=200; total time=   0.6s
    [CV] END ......max_depth=3, max_features=8, n_estimators=300; total time=   0.8s
    [CV] END .....max_depth=3, max_features=16, n_estimators=300; total time=   1.3s
    [CV] END ...max_depth=3, max_features=log2, n_estimators=200; total time=   0.5s
    [CV] END ...max_depth=3, max_features=log2, n_estimators=300; total time=   0.6s
    [CV] END ...max_depth=3, max_features=None, n_estimators=200; total time=   1.1s
    [CV] END ...max_depth=5, max_features=sqrt, n_estimators=100; total time=   0.3s
    [CV] END ...max_depth=5, max_features=sqrt, n_estimators=200; total time=   0.6s
    [CV] END ...max_depth=5, max_features=sqrt, n_estimators=200; total time=   0.6s
    [CV] END ......max_depth=5, max_features=2, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=5, max_features=2, n_estimators=200; total time=   0.4s
    [CV] END ......max_depth=5, max_features=2, n_estimators=300; total time=   0.6s
    [CV] END ......max_depth=5, max_features=4, n_estimators=300; total time=   0.7s
    [CV] END ......max_depth=5, max_features=8, n_estimators=100; total time=   0.3s
    [CV] END ......max_depth=5, max_features=8, n_estimators=300; total time=   1.0s
    [CV] END .....max_depth=5, max_features=16, n_estimators=200; total time=   1.0s
    [CV] END ...max_depth=5, max_features=log2, n_estimators=100; total time=   0.3s
    [CV] END ...max_depth=5, max_features=log2, n_estimators=100; total time=   0.3s
    [CV] END ...max_depth=5, max_features=log2, n_estimators=200; total time=   0.5s
    [CV] END ...max_depth=5, max_features=log2, n_estimators=300; total time=   0.7s
    [CV] END ...max_depth=5, max_features=None, n_estimators=100; total time=   0.7s
    [CV] END ...max_depth=5, max_features=None, n_estimators=300; total time=   2.2s
    [CV] END ..max_depth=10, max_features=sqrt, n_estimators=300; total time=   1.2s
    [CV] END .....max_depth=10, max_features=2, n_estimators=200; total time=   0.6s
    [CV] END .....max_depth=10, max_features=2, n_estimators=300; total time=   0.9s
    [CV] END .....max_depth=10, max_features=4, n_estimators=300; total time=   1.2s
    [CV] END .....max_depth=10, max_features=8, n_estimators=100; total time=   0.5s
    [CV] END .....max_depth=10, max_features=8, n_estimators=300; total time=   1.6s
    [CV] END ....max_depth=10, max_features=16, n_estimators=200; total time=   1.5s
    [CV] END ....max_depth=10, max_features=16, n_estimators=300; total time=   2.5s
    [CV] END ..max_depth=10, max_features=None, n_estimators=100; total time=   1.2s
    [CV] END ..max_depth=10, max_features=None, n_estimators=300; total time=   3.6s
    [CV] END ..max_depth=20, max_features=sqrt, n_estimators=200; total time=   1.0s
    [CV] END .....max_depth=20, max_features=2, n_estimators=100; total time=   0.3s
    [CV] END .....max_depth=20, max_features=2, n_estimators=100; total time=   0.3s
    [CV] END .....max_depth=20, max_features=2, n_estimators=200; total time=   0.8s
    [CV] END .....max_depth=20, max_features=2, n_estimators=300; total time=   1.4s
    [CV] END .....max_depth=20, max_features=4, n_estimators=300; total time=   1.6s
    [CV] END .....max_depth=20, max_features=8, n_estimators=200; total time=   1.1s
    [CV] END .....max_depth=20, max_features=8, n_estimators=300; total time=   1.8s
    [CV] END ....max_depth=20, max_features=16, n_estimators=200; total time=   1.9s
    [CV] END ..max_depth=20, max_features=log2, n_estimators=100; total time=   0.7s
    [CV] END ..max_depth=20, max_features=log2, n_estimators=200; total time=   1.1s
    [CV] END ..max_depth=20, max_features=log2, n_estimators=300; total time=   2.1s
    [CV] END ..max_depth=20, max_features=None, n_estimators=100; total time=   1.7s
    [CV] END ..max_depth=20, max_features=None, n_estimators=300; total time=   3.6s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=100, subsample=0.5; total time=   0.3s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=100, subsample=0.9; total time=   0.4s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=100, subsample=1; total time=   0.5s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=200, subsample=0.7; total time=   0.7s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=200, subsample=0.9; total time=   0.8s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=500, subsample=0.5; total time=   1.5s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=500, subsample=0.7; total time=   1.9s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=500, subsample=1; total time=   2.6s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.9; total time=   0.6s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=200, subsample=0.5; total time=   0.9s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=200, subsample=0.9; total time=   1.2s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=200, subsample=1; total time=   1.3s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=500, subsample=0.7; total time=   3.1s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=500, subsample=0.9; total time=   3.9s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=100, subsample=0.5; total time=   1.1s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=100, subsample=0.7; total time=   1.3s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=100, subsample=0.9; total time=   1.4s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=200, subsample=0.5; total time=   1.9s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=200, subsample=0.7; total time=   2.1s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=200, subsample=1; total time=   2.3s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=500, subsample=0.7; total time=   4.2s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=500, subsample=0.9; total time=   6.9s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=100, subsample=0.5; total time=   2.5s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=100, subsample=0.7; total time=   3.0s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=100, subsample=0.9; total time=   3.7s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=200, subsample=0.5; total time=   5.4s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=200, subsample=0.9; total time=   7.9s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=200, subsample=1; total time=   6.8s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=500, subsample=0.7; total time=  12.9s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=500, subsample=1; total time=  15.4s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=100, subsample=0.5; total time=   7.2s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=100, subsample=0.7; total time=   9.6s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=100, subsample=1; total time=   4.6s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=200, subsample=0.5; total time=  13.9s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=200, subsample=0.9; total time=  20.9s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=200, subsample=1; total time=   8.8s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=500, subsample=0.7; total time=  47.6s
    [CV] END max_depth=None, max_features=sqrt, n_estimators=100; total time=   0.6s
    [CV] END max_depth=None, max_features=sqrt, n_estimators=200; total time=   1.2s
    [CV] END ...max_depth=None, max_features=2, n_estimators=100; total time=   0.4s
    [CV] END ...max_depth=None, max_features=2, n_estimators=100; total time=   0.4s
    [CV] END ...max_depth=None, max_features=2, n_estimators=300; total time=   1.1s
    [CV] END ...max_depth=None, max_features=4, n_estimators=200; total time=   1.0s
    [CV] END ...max_depth=None, max_features=4, n_estimators=300; total time=   1.3s
    [CV] END ...max_depth=None, max_features=8, n_estimators=300; total time=   1.8s
    [CV] END ..max_depth=None, max_features=16, n_estimators=200; total time=   1.9s
    [CV] END ..max_depth=None, max_features=16, n_estimators=300; total time=   2.9s
    [CV] END max_depth=None, max_features=None, n_estimators=100; total time=   1.5s
    [CV] END max_depth=None, max_features=None, n_estimators=200; total time=   2.7s
    [CV] END ...max_depth=2, max_features=sqrt, n_estimators=100; total time=   0.2s
    [CV] END ...max_depth=2, max_features=sqrt, n_estimators=200; total time=   0.4s
    [CV] END ...max_depth=2, max_features=sqrt, n_estimators=200; total time=   0.4s
    [CV] END ...max_depth=2, max_features=sqrt, n_estimators=300; total time=   0.6s
    [CV] END ......max_depth=2, max_features=2, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=2, max_features=2, n_estimators=200; total time=   0.3s
    [CV] END ......max_depth=2, max_features=2, n_estimators=300; total time=   0.5s
    [CV] END ......max_depth=2, max_features=4, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=2, max_features=4, n_estimators=200; total time=   0.4s
    [CV] END ......max_depth=2, max_features=8, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=2, max_features=8, n_estimators=200; total time=   0.4s
    [CV] END ......max_depth=2, max_features=8, n_estimators=300; total time=   0.7s
    [CV] END .....max_depth=2, max_features=16, n_estimators=200; total time=   0.6s
    [CV] END ...max_depth=2, max_features=log2, n_estimators=100; total time=   0.2s
    [CV] END ...max_depth=2, max_features=log2, n_estimators=200; total time=   0.5s
    [CV] END ...max_depth=2, max_features=log2, n_estimators=300; total time=   0.6s
    [CV] END ...max_depth=2, max_features=None, n_estimators=300; total time=   1.2s
    [CV] END ...max_depth=3, max_features=sqrt, n_estimators=200; total time=   0.5s
    [CV] END ...max_depth=3, max_features=sqrt, n_estimators=300; total time=   0.7s
    [CV] END ......max_depth=3, max_features=2, n_estimators=300; total time=   0.5s
    [CV] END ......max_depth=3, max_features=4, n_estimators=200; total time=   0.5s
    [CV] END ......max_depth=3, max_features=4, n_estimators=300; total time=   0.7s
    [CV] END ......max_depth=3, max_features=8, n_estimators=300; total time=   0.9s
    [CV] END .....max_depth=3, max_features=16, n_estimators=200; total time=   0.8s
    [CV] END ...max_depth=3, max_features=log2, n_estimators=100; total time=   0.2s
    [CV] END ...max_depth=3, max_features=log2, n_estimators=100; total time=   0.3s
    [CV] END ...max_depth=3, max_features=log2, n_estimators=200; total time=   0.5s
    [CV] END ...max_depth=3, max_features=log2, n_estimators=300; total time=   0.7s
    [CV] END ...max_depth=3, max_features=None, n_estimators=200; total time=   1.0s
    [CV] END ...max_depth=3, max_features=None, n_estimators=300; total time=   1.6s
    [CV] END ......max_depth=5, max_features=2, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=5, max_features=2, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=5, max_features=2, n_estimators=200; total time=   0.4s
    [CV] END ......max_depth=5, max_features=4, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=5, max_features=4, n_estimators=200; total time=   0.5s
    [CV] END ......max_depth=5, max_features=4, n_estimators=300; total time=   0.7s
    [CV] END ......max_depth=5, max_features=8, n_estimators=200; total time=   0.7s
    [CV] END .....max_depth=5, max_features=16, n_estimators=100; total time=   0.5s
    [CV] END .....max_depth=5, max_features=16, n_estimators=200; total time=   1.0s
    [CV] END ...max_depth=5, max_features=log2, n_estimators=100; total time=   0.3s
    [CV] END ...max_depth=5, max_features=log2, n_estimators=100; total time=   0.3s
    [CV] END ...max_depth=5, max_features=log2, n_estimators=200; total time=   0.5s
    [CV] END ...max_depth=5, max_features=log2, n_estimators=300; total time=   0.7s
    [CV] END ...max_depth=5, max_features=None, n_estimators=200; total time=   1.4s
    [CV] END ...max_depth=5, max_features=None, n_estimators=300; total time=   2.1s
    [CV] END .....max_depth=10, max_features=2, n_estimators=100; total time=   0.3s
    [CV] END .....max_depth=10, max_features=2, n_estimators=200; total time=   0.6s
    [CV] END .....max_depth=10, max_features=2, n_estimators=300; total time=   0.9s
    [CV] END .....max_depth=10, max_features=4, n_estimators=200; total time=   0.8s
    [CV] END .....max_depth=10, max_features=4, n_estimators=300; total time=   1.1s
    [CV] END .....max_depth=10, max_features=8, n_estimators=300; total time=   1.5s
    [CV] END ....max_depth=10, max_features=16, n_estimators=200; total time=   1.6s
    [CV] END ..max_depth=10, max_features=log2, n_estimators=100; total time=   0.4s
    [CV] END ..max_depth=10, max_features=log2, n_estimators=100; total time=   0.4s
    [CV] END ..max_depth=10, max_features=log2, n_estimators=200; total time=   0.7s
    [CV] END ..max_depth=10, max_features=log2, n_estimators=200; total time=   0.8s
    [CV] END ..max_depth=10, max_features=None, n_estimators=100; total time=   1.3s
    [CV] END ..max_depth=10, max_features=None, n_estimators=200; total time=   2.4s
    [CV] END ..max_depth=20, max_features=sqrt, n_estimators=100; total time=   0.4s
    [CV] END ..max_depth=20, max_features=sqrt, n_estimators=100; total time=   0.5s
    [CV] END ..max_depth=20, max_features=sqrt, n_estimators=200; total time=   1.0s
    [CV] END ..max_depth=20, max_features=sqrt, n_estimators=300; total time=   1.5s
    [CV] END .....max_depth=20, max_features=2, n_estimators=300; total time=   1.2s
    [CV] END .....max_depth=20, max_features=4, n_estimators=200; total time=   1.3s
    [CV] END .....max_depth=20, max_features=4, n_estimators=300; total time=   1.4s
    [CV] END .....max_depth=20, max_features=8, n_estimators=300; total time=   1.8s
    [CV] END ....max_depth=20, max_features=16, n_estimators=100; total time=   1.0s
    [CV] END ....max_depth=20, max_features=16, n_estimators=300; total time=   3.0s
    [CV] END ..max_depth=20, max_features=log2, n_estimators=200; total time=   0.9s
    [CV] END ..max_depth=20, max_features=None, n_estimators=100; total time=   2.1s
    [CV] END ..max_depth=20, max_features=None, n_estimators=200; total time=   3.0s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=100, subsample=0.5; total time=   0.3s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=100, subsample=0.7; total time=   0.4s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=100, subsample=1; total time=   0.5s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=200, subsample=0.7; total time=   0.8s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=200, subsample=0.9; total time=   0.8s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=500, subsample=0.5; total time=   1.6s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=500, subsample=0.9; total time=   2.2s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=500, subsample=1; total time=   2.3s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=100, subsample=1; total time=   0.6s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=200, subsample=0.5; total time=   0.9s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=200, subsample=0.9; total time=   1.3s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=200, subsample=1; total time=   1.3s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=500, subsample=0.7; total time=   3.1s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=500, subsample=1; total time=   4.0s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=100, subsample=0.5; total time=   1.1s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=100, subsample=0.7; total time=   1.3s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=100, subsample=0.9; total time=   1.6s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=200, subsample=0.5; total time=   1.8s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=200, subsample=0.9; total time=   2.4s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=200, subsample=1; total time=   2.1s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=500, subsample=0.7; total time=   4.2s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=500, subsample=0.9; total time=   6.6s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=100, subsample=0.5; total time=   2.5s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=100, subsample=0.7; total time=   3.2s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=100, subsample=1; total time=   3.3s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=200, subsample=0.5; total time=   5.1s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=200, subsample=0.7; total time=   6.8s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=200, subsample=1; total time=   6.9s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=500, subsample=0.7; total time=  13.0s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=500, subsample=0.9; total time=  15.2s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=100, subsample=0.5; total time=   7.3s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=100, subsample=0.7; total time=   9.5s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=100, subsample=1; total time=   4.6s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=200, subsample=0.5; total time=  14.2s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=200, subsample=0.9; total time=  22.4s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=500, subsample=0.5; total time=  33.7s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=500, subsample=0.7; total time=  46.4s
    [CV] END max_depth=None, max_features=sqrt, n_estimators=100; total time=   0.6s
    [CV] END max_depth=None, max_features=sqrt, n_estimators=300; total time=   1.6s
    [CV] END ...max_depth=None, max_features=2, n_estimators=200; total time=   0.7s
    [CV] END ...max_depth=None, max_features=4, n_estimators=100; total time=   0.5s
    [CV] END ...max_depth=None, max_features=4, n_estimators=100; total time=   0.4s
    [CV] END ...max_depth=None, max_features=4, n_estimators=200; total time=   1.0s
    [CV] END ...max_depth=None, max_features=8, n_estimators=100; total time=   0.6s
    [CV] END ...max_depth=None, max_features=8, n_estimators=100; total time=   0.6s
    [CV] END ...max_depth=None, max_features=8, n_estimators=300; total time=   1.9s
    [CV] END ..max_depth=None, max_features=16, n_estimators=200; total time=   2.0s
    [CV] END max_depth=None, max_features=log2, n_estimators=100; total time=   0.5s
    [CV] END max_depth=None, max_features=log2, n_estimators=100; total time=   0.4s
    [CV] END max_depth=None, max_features=log2, n_estimators=200; total time=   0.9s
    [CV] END max_depth=None, max_features=log2, n_estimators=300; total time=   1.4s
    [CV] END max_depth=None, max_features=None, n_estimators=100; total time=   1.6s
    [CV] END max_depth=None, max_features=None, n_estimators=300; total time=   4.1s
    [CV] END ......max_depth=2, max_features=2, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=2, max_features=2, n_estimators=200; total time=   0.3s
    [CV] END ......max_depth=2, max_features=2, n_estimators=300; total time=   0.5s
    [CV] END ......max_depth=2, max_features=4, n_estimators=200; total time=   0.4s
    [CV] END ......max_depth=2, max_features=4, n_estimators=300; total time=   0.6s
    [CV] END ......max_depth=2, max_features=8, n_estimators=300; total time=   0.7s
    [CV] END .....max_depth=2, max_features=16, n_estimators=200; total time=   0.7s
    [CV] END ...max_depth=2, max_features=log2, n_estimators=100; total time=   0.2s
    [CV] END ...max_depth=2, max_features=log2, n_estimators=100; total time=   0.2s
    [CV] END ...max_depth=2, max_features=log2, n_estimators=200; total time=   0.4s
    [CV] END ...max_depth=2, max_features=log2, n_estimators=300; total time=   0.6s
    [CV] END ...max_depth=2, max_features=None, n_estimators=200; total time=   0.9s
    [CV] END ...max_depth=3, max_features=sqrt, n_estimators=100; total time=   0.3s
    [CV] END ...max_depth=3, max_features=sqrt, n_estimators=100; total time=   0.3s
    [CV] END ...max_depth=3, max_features=sqrt, n_estimators=200; total time=   0.4s
    [CV] END ...max_depth=3, max_features=sqrt, n_estimators=300; total time=   0.7s
    [CV] END ......max_depth=3, max_features=2, n_estimators=300; total time=   0.6s
    [CV] END ......max_depth=3, max_features=4, n_estimators=200; total time=   0.4s
    [CV] END ......max_depth=3, max_features=4, n_estimators=300; total time=   0.6s
    [CV] END ......max_depth=3, max_features=8, n_estimators=300; total time=   1.0s
    [CV] END .....max_depth=3, max_features=16, n_estimators=200; total time=   0.8s
    [CV] END ...max_depth=3, max_features=log2, n_estimators=100; total time=   0.2s
    [CV] END ...max_depth=3, max_features=log2, n_estimators=100; total time=   0.2s
    [CV] END ...max_depth=3, max_features=log2, n_estimators=200; total time=   0.5s
    [CV] END ...max_depth=3, max_features=log2, n_estimators=300; total time=   0.7s
    [CV] END ...max_depth=3, max_features=None, n_estimators=100; total time=   0.5s
    [CV] END ...max_depth=3, max_features=None, n_estimators=300; total time=   1.6s
    [CV] END ...max_depth=5, max_features=sqrt, n_estimators=300; total time=   0.8s
    [CV] END ......max_depth=5, max_features=2, n_estimators=200; total time=   0.4s
    [CV] END ......max_depth=5, max_features=2, n_estimators=300; total time=   0.6s
    [CV] END ......max_depth=5, max_features=4, n_estimators=300; total time=   0.8s
    [CV] END ......max_depth=5, max_features=8, n_estimators=200; total time=   0.7s
    [CV] END .....max_depth=5, max_features=16, n_estimators=100; total time=   0.5s
    [CV] END .....max_depth=5, max_features=16, n_estimators=100; total time=   0.5s
    [CV] END .....max_depth=5, max_features=16, n_estimators=300; total time=   1.5s
    [CV] END ...max_depth=5, max_features=log2, n_estimators=200; total time=   0.5s
    [CV] END ...max_depth=5, max_features=None, n_estimators=100; total time=   0.7s
    [CV] END ...max_depth=5, max_features=None, n_estimators=200; total time=   1.4s
    [CV] END ..max_depth=10, max_features=sqrt, n_estimators=100; total time=   0.4s
    [CV] END ..max_depth=10, max_features=sqrt, n_estimators=100; total time=   0.4s
    [CV] END ..max_depth=10, max_features=sqrt, n_estimators=200; total time=   0.8s
    [CV] END ..max_depth=10, max_features=sqrt, n_estimators=300; total time=   1.2s
    [CV] END .....max_depth=10, max_features=2, n_estimators=300; total time=   0.9s
    [CV] END .....max_depth=10, max_features=4, n_estimators=200; total time=   0.8s
    [CV] END .....max_depth=10, max_features=8, n_estimators=100; total time=   0.6s
    [CV] END .....max_depth=10, max_features=8, n_estimators=200; total time=   1.0s
    [CV] END ....max_depth=10, max_features=16, n_estimators=100; total time=   0.7s
    [CV] END ....max_depth=10, max_features=16, n_estimators=100; total time=   0.8s
    [CV] END ....max_depth=10, max_features=16, n_estimators=300; total time=   2.4s
    [CV] END ..max_depth=10, max_features=log2, n_estimators=300; total time=   1.2s
    [CV] END ..max_depth=10, max_features=None, n_estimators=100; total time=   1.2s
    [CV] END ..max_depth=10, max_features=None, n_estimators=300; total time=   3.6s
    [CV] END ..max_depth=20, max_features=sqrt, n_estimators=300; total time=   1.4s
    [CV] END .....max_depth=20, max_features=2, n_estimators=100; total time=   0.4s
    [CV] END .....max_depth=20, max_features=2, n_estimators=300; total time=   1.2s
    [CV] END .....max_depth=20, max_features=4, n_estimators=100; total time=   0.7s
    [CV] END .....max_depth=20, max_features=4, n_estimators=300; total time=   1.7s
    [CV] END .....max_depth=20, max_features=8, n_estimators=200; total time=   1.3s
    [CV] END ....max_depth=20, max_features=16, n_estimators=100; total time=   0.9s
    [CV] END ....max_depth=20, max_features=16, n_estimators=200; total time=   2.1s
    [CV] END ....max_depth=20, max_features=16, n_estimators=300; total time=   3.5s
    [CV] END ..max_depth=20, max_features=None, n_estimators=100; total time=   1.8s
    [CV] END ..max_depth=20, max_features=None, n_estimators=300; total time=   3.7s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=100, subsample=0.7; total time=   0.4s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=100, subsample=1; total time=   0.4s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=200, subsample=0.5; total time=   0.7s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=200, subsample=0.9; total time=   0.9s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=200, subsample=1; total time=   0.9s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=500, subsample=0.7; total time=   1.9s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=500, subsample=0.9; total time=   2.3s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.5; total time=   0.5s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.7; total time=   0.5s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.9; total time=   0.6s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=100, subsample=1; total time=   0.7s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=200, subsample=0.7; total time=   1.1s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=200, subsample=1; total time=   1.3s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=500, subsample=0.5; total time=   2.5s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=500, subsample=0.9; total time=   3.6s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=500, subsample=1; total time=   4.7s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=100, subsample=1; total time=   1.6s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=200, subsample=0.7; total time=   2.1s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=200, subsample=1; total time=   2.3s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=500, subsample=0.5; total time=   3.4s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=500, subsample=0.9; total time=   6.7s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=500, subsample=1; total time=   5.4s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=100, subsample=0.9; total time=   3.6s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=100, subsample=1; total time=   3.2s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=200, subsample=0.7; total time=   6.7s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=200, subsample=1; total time=   7.3s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=500, subsample=0.5; total time=  10.7s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=500, subsample=0.9; total time=  15.0s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=500, subsample=1; total time=  15.5s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=100, subsample=0.9; total time=   9.6s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=100, subsample=1; total time=   4.4s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=200, subsample=0.5; total time=  14.2s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=200, subsample=0.9; total time=  22.0s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=500, subsample=0.5; total time=  34.0s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=500, subsample=0.7; total time=  45.9s
    [CV] END max_depth=None, max_features=sqrt, n_estimators=100; total time=   0.6s
    [CV] END max_depth=None, max_features=sqrt, n_estimators=300; total time=   1.6s
    [CV] END ...max_depth=None, max_features=2, n_estimators=200; total time=   0.8s
    [CV] END ...max_depth=None, max_features=4, n_estimators=100; total time=   0.5s
    [CV] END ...max_depth=None, max_features=4, n_estimators=200; total time=   1.0s
    [CV] END ...max_depth=None, max_features=4, n_estimators=300; total time=   1.3s
    [CV] END ...max_depth=None, max_features=8, n_estimators=200; total time=   1.2s
    [CV] END ..max_depth=None, max_features=16, n_estimators=100; total time=   1.0s
    [CV] END ..max_depth=None, max_features=16, n_estimators=200; total time=   1.9s
    [CV] END max_depth=None, max_features=log2, n_estimators=100; total time=   0.4s
    [CV] END max_depth=None, max_features=log2, n_estimators=100; total time=   0.5s
    [CV] END max_depth=None, max_features=log2, n_estimators=200; total time=   0.9s
    [CV] END max_depth=None, max_features=log2, n_estimators=300; total time=   1.3s
    [CV] END max_depth=None, max_features=None, n_estimators=200; total time=   2.8s
    [CV] END max_depth=None, max_features=None, n_estimators=300; total time=   4.2s
    [CV] END ......max_depth=2, max_features=8, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=2, max_features=8, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=2, max_features=8, n_estimators=200; total time=   0.5s
    [CV] END .....max_depth=2, max_features=16, n_estimators=100; total time=   0.3s
    [CV] END .....max_depth=2, max_features=16, n_estimators=200; total time=   0.6s
    [CV] END ...max_depth=2, max_features=log2, n_estimators=100; total time=   0.2s
    [CV] END ...max_depth=2, max_features=log2, n_estimators=100; total time=   0.2s
    [CV] END ...max_depth=2, max_features=log2, n_estimators=200; total time=   0.4s
    [CV] END ...max_depth=2, max_features=None, n_estimators=100; total time=   0.4s
    [CV] END ...max_depth=2, max_features=None, n_estimators=200; total time=   0.8s
    [CV] END ...max_depth=2, max_features=None, n_estimators=300; total time=   1.2s
    [CV] END ......max_depth=3, max_features=2, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=3, max_features=2, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=3, max_features=2, n_estimators=200; total time=   0.4s
    [CV] END ......max_depth=3, max_features=4, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=3, max_features=4, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=3, max_features=4, n_estimators=300; total time=   0.7s
    [CV] END ......max_depth=3, max_features=8, n_estimators=200; total time=   0.7s
    [CV] END .....max_depth=3, max_features=16, n_estimators=100; total time=   0.5s
    [CV] END .....max_depth=3, max_features=16, n_estimators=200; total time=   0.8s
    [CV] END .....max_depth=3, max_features=16, n_estimators=300; total time=   1.3s
    [CV] END ...max_depth=3, max_features=None, n_estimators=100; total time=   0.6s
    [CV] END ...max_depth=3, max_features=None, n_estimators=200; total time=   1.0s
    [CV] END ...max_depth=5, max_features=sqrt, n_estimators=100; total time=   0.3s
    [CV] END ...max_depth=5, max_features=sqrt, n_estimators=100; total time=   0.3s
    [CV] END ...max_depth=5, max_features=sqrt, n_estimators=200; total time=   0.6s
    [CV] END ...max_depth=5, max_features=sqrt, n_estimators=300; total time=   0.8s
    [CV] END ......max_depth=5, max_features=2, n_estimators=300; total time=   0.7s
    [CV] END ......max_depth=5, max_features=4, n_estimators=200; total time=   0.5s
    [CV] END ......max_depth=5, max_features=8, n_estimators=100; total time=   0.3s
    [CV] END ......max_depth=5, max_features=8, n_estimators=200; total time=   0.7s
    [CV] END ......max_depth=5, max_features=8, n_estimators=300; total time=   1.0s
    [CV] END .....max_depth=5, max_features=16, n_estimators=300; total time=   1.5s
    [CV] END ...max_depth=5, max_features=log2, n_estimators=200; total time=   0.5s
    [CV] END ...max_depth=5, max_features=log2, n_estimators=300; total time=   0.7s
    [CV] END ...max_depth=5, max_features=None, n_estimators=200; total time=   1.5s
    [CV] END ..max_depth=10, max_features=sqrt, n_estimators=100; total time=   0.4s
    [CV] END ..max_depth=10, max_features=sqrt, n_estimators=100; total time=   0.4s
    [CV] END ..max_depth=10, max_features=sqrt, n_estimators=200; total time=   0.8s
    [CV] END ..max_depth=10, max_features=sqrt, n_estimators=300; total time=   1.2s
    [CV] END .....max_depth=10, max_features=2, n_estimators=300; total time=   0.9s
    [CV] END .....max_depth=10, max_features=4, n_estimators=200; total time=   0.8s
    [CV] END .....max_depth=10, max_features=8, n_estimators=100; total time=   0.5s
    [CV] END .....max_depth=10, max_features=8, n_estimators=200; total time=   1.0s
    [CV] END ....max_depth=10, max_features=16, n_estimators=100; total time=   0.8s
    [CV] END ....max_depth=10, max_features=16, n_estimators=100; total time=   1.0s
    [CV] END ....max_depth=10, max_features=16, n_estimators=300; total time=   2.5s
    [CV] END ..max_depth=10, max_features=log2, n_estimators=300; total time=   1.2s
    [CV] END ..max_depth=10, max_features=None, n_estimators=200; total time=   2.5s
    [CV] END ..max_depth=10, max_features=None, n_estimators=300; total time=   3.6s
    [CV] END .....max_depth=20, max_features=2, n_estimators=100; total time=   0.4s
    [CV] END .....max_depth=20, max_features=2, n_estimators=200; total time=   0.8s
    [CV] END .....max_depth=20, max_features=4, n_estimators=100; total time=   0.5s
    [CV] END .....max_depth=20, max_features=4, n_estimators=200; total time=   1.3s
    [CV] END .....max_depth=20, max_features=4, n_estimators=300; total time=   1.5s
    [CV] END .....max_depth=20, max_features=8, n_estimators=300; total time=   1.9s
    [CV] END ....max_depth=20, max_features=16, n_estimators=200; total time=   2.1s
    [CV] END ..max_depth=20, max_features=log2, n_estimators=100; total time=   0.7s
    [CV] END ..max_depth=20, max_features=log2, n_estimators=100; total time=   0.5s
    [CV] END ..max_depth=20, max_features=log2, n_estimators=200; total time=   1.1s
    [CV] END ..max_depth=20, max_features=log2, n_estimators=300; total time=   2.1s
    [CV] END ..max_depth=20, max_features=None, n_estimators=200; total time=   3.0s
    [CV] END ..max_depth=20, max_features=None, n_estimators=300; total time=   3.2s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=100, subsample=0.7; total time=   0.4s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=100, subsample=0.9; total time=   0.5s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=200, subsample=0.5; total time=   0.7s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=200, subsample=0.7; total time=   0.9s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=200, subsample=1; total time=   0.9s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=500, subsample=0.7; total time=   1.9s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=500, subsample=1; total time=   2.4s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.5; total time=   0.4s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.7; total time=   0.5s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.9; total time=   0.6s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=200, subsample=0.5; total time=   0.9s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=200, subsample=0.7; total time=   1.2s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=200, subsample=1; total time=   1.3s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=500, subsample=0.7; total time=   3.0s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=500, subsample=0.9; total time=   3.9s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=100, subsample=0.5; total time=   0.9s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=100, subsample=0.7; total time=   1.4s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=100, subsample=0.9; total time=   1.3s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=100, subsample=1; total time=   1.6s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=200, subsample=0.7; total time=   2.1s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=200, subsample=0.9; total time=   2.2s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=500, subsample=0.5; total time=   3.5s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=500, subsample=0.9; total time=   6.4s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=500, subsample=1; total time=   5.4s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=100, subsample=0.9; total time=   3.4s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=100, subsample=1; total time=   3.2s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=200, subsample=0.7; total time=   6.7s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=200, subsample=0.9; total time=   7.7s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=500, subsample=0.5; total time=  10.8s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=500, subsample=0.9; total time=  14.8s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=500, subsample=1; total time=  15.5s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=100, subsample=0.9; total time=  10.0s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=100, subsample=1; total time=   4.4s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=200, subsample=0.7; total time=  18.8s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=200, subsample=0.9; total time=  21.7s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=500, subsample=0.5; total time=  34.4s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=500, subsample=0.9; total time=  53.6s
    [CV] END max_depth=None, max_features=sqrt, n_estimators=100; total time=   0.5s
    [CV] END max_depth=None, max_features=sqrt, n_estimators=300; total time=   1.6s
    [CV] END ...max_depth=None, max_features=2, n_estimators=200; total time=   0.8s
    [CV] END ...max_depth=None, max_features=4, n_estimators=100; total time=   0.5s
    [CV] END ...max_depth=None, max_features=4, n_estimators=100; total time=   0.5s
    [CV] END ...max_depth=None, max_features=4, n_estimators=300; total time=   1.4s
    [CV] END ...max_depth=None, max_features=8, n_estimators=100; total time=   0.6s
    [CV] END ...max_depth=None, max_features=8, n_estimators=300; total time=   1.9s
    [CV] END ..max_depth=None, max_features=16, n_estimators=200; total time=   2.0s
    [CV] END ..max_depth=None, max_features=16, n_estimators=300; total time=   2.9s
    [CV] END max_depth=None, max_features=None, n_estimators=100; total time=   1.5s
    [CV] END max_depth=None, max_features=None, n_estimators=300; total time=   4.1s
    [CV] END ...max_depth=2, max_features=sqrt, n_estimators=300; total time=   0.6s
    [CV] END ......max_depth=2, max_features=2, n_estimators=300; total time=   0.5s
    [CV] END ......max_depth=2, max_features=4, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=2, max_features=4, n_estimators=200; total time=   0.4s
    [CV] END ......max_depth=2, max_features=4, n_estimators=300; total time=   0.5s
    [CV] END ......max_depth=2, max_features=8, n_estimators=200; total time=   0.5s
    [CV] END .....max_depth=2, max_features=16, n_estimators=100; total time=   0.3s
    [CV] END .....max_depth=2, max_features=16, n_estimators=100; total time=   0.3s
    [CV] END .....max_depth=2, max_features=16, n_estimators=300; total time=   0.9s
    [CV] END ...max_depth=2, max_features=log2, n_estimators=200; total time=   0.3s
    [CV] END ...max_depth=2, max_features=None, n_estimators=100; total time=   0.4s
    [CV] END ...max_depth=2, max_features=None, n_estimators=200; total time=   0.8s
    [CV] END ...max_depth=2, max_features=None, n_estimators=300; total time=   1.2s
    [CV] END ......max_depth=3, max_features=2, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=3, max_features=2, n_estimators=200; total time=   0.4s
    [CV] END ......max_depth=3, max_features=2, n_estimators=300; total time=   0.5s
    [CV] END ......max_depth=3, max_features=4, n_estimators=200; total time=   0.5s
    [CV] END ......max_depth=3, max_features=8, n_estimators=100; total time=   0.3s
    [CV] END ......max_depth=3, max_features=8, n_estimators=200; total time=   0.6s
    [CV] END .....max_depth=3, max_features=16, n_estimators=100; total time=   0.4s
    [CV] END .....max_depth=3, max_features=16, n_estimators=100; total time=   0.4s
    [CV] END .....max_depth=3, max_features=16, n_estimators=200; total time=   0.8s
    [CV] END ...max_depth=3, max_features=log2, n_estimators=100; total time=   0.3s
    [CV] END ...max_depth=3, max_features=log2, n_estimators=200; total time=   0.5s
    [CV] END ...max_depth=3, max_features=log2, n_estimators=300; total time=   0.7s
    [CV] END ...max_depth=3, max_features=None, n_estimators=200; total time=   1.0s
    [CV] END ...max_depth=5, max_features=sqrt, n_estimators=100; total time=   0.3s
    [CV] END ...max_depth=5, max_features=sqrt, n_estimators=100; total time=   0.3s
    [CV] END ...max_depth=5, max_features=sqrt, n_estimators=200; total time=   0.6s
    [CV] END ...max_depth=5, max_features=sqrt, n_estimators=300; total time=   0.8s
    [CV] END ......max_depth=5, max_features=2, n_estimators=300; total time=   0.6s
    [CV] END ......max_depth=5, max_features=4, n_estimators=100; total time=   0.3s
    [CV] END ......max_depth=5, max_features=4, n_estimators=300; total time=   0.8s
    [CV] END ......max_depth=5, max_features=8, n_estimators=200; total time=   0.7s
    [CV] END ......max_depth=5, max_features=8, n_estimators=300; total time=   1.0s
    [CV] END .....max_depth=5, max_features=16, n_estimators=200; total time=   1.0s
    [CV] END ...max_depth=5, max_features=log2, n_estimators=100; total time=   0.3s
    [CV] END ...max_depth=5, max_features=log2, n_estimators=200; total time=   0.5s
    [CV] END ...max_depth=5, max_features=log2, n_estimators=300; total time=   0.8s
    [CV] END ...max_depth=5, max_features=None, n_estimators=200; total time=   1.4s
    [CV] END ...max_depth=5, max_features=None, n_estimators=300; total time=   2.1s
    [CV] END .....max_depth=10, max_features=2, n_estimators=100; total time=   0.3s
    [CV] END .....max_depth=10, max_features=2, n_estimators=100; total time=   0.3s
    [CV] END .....max_depth=10, max_features=2, n_estimators=200; total time=   0.6s
    [CV] END .....max_depth=10, max_features=4, n_estimators=100; total time=   0.4s
    [CV] END .....max_depth=10, max_features=4, n_estimators=100; total time=   0.4s
    [CV] END .....max_depth=10, max_features=4, n_estimators=300; total time=   1.2s
    [CV] END .....max_depth=10, max_features=8, n_estimators=200; total time=   1.0s
    [CV] END .....max_depth=10, max_features=8, n_estimators=300; total time=   1.5s
    [CV] END ....max_depth=10, max_features=16, n_estimators=200; total time=   1.7s
    [CV] END ..max_depth=10, max_features=log2, n_estimators=100; total time=   0.4s
    [CV] END ..max_depth=10, max_features=log2, n_estimators=200; total time=   0.7s
    [CV] END ..max_depth=10, max_features=log2, n_estimators=300; total time=   1.1s
    [CV] END ..max_depth=10, max_features=None, n_estimators=100; total time=   1.2s
    [CV] END ..max_depth=10, max_features=None, n_estimators=300; total time=   3.3s
    [CV] END ..max_depth=20, max_features=sqrt, n_estimators=200; total time=   1.0s
    [CV] END ..max_depth=20, max_features=sqrt, n_estimators=300; total time=   1.4s
    [CV] END .....max_depth=20, max_features=2, n_estimators=300; total time=   1.4s
    [CV] END .....max_depth=20, max_features=4, n_estimators=300; total time=   1.7s
    [CV] END .....max_depth=20, max_features=8, n_estimators=200; total time=   1.2s
    [CV] END .....max_depth=20, max_features=8, n_estimators=300; total time=   1.8s
    [CV] END ....max_depth=20, max_features=16, n_estimators=300; total time=   3.1s
    [CV] END ..max_depth=20, max_features=log2, n_estimators=200; total time=   1.0s
    [CV] END ..max_depth=20, max_features=log2, n_estimators=300; total time=   2.1s
    [CV] END ..max_depth=20, max_features=None, n_estimators=200; total time=   2.8s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=100, subsample=0.5; total time=   0.3s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=100, subsample=0.7; total time=   0.4s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=100, subsample=1; total time=   0.4s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=200, subsample=0.5; total time=   0.6s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=200, subsample=0.9; total time=   0.8s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=500, subsample=0.5; total time=   1.6s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=500, subsample=0.7; total time=   1.9s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=500, subsample=1; total time=   2.3s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.7; total time=   0.5s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=100, subsample=1; total time=   0.6s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=200, subsample=0.5; total time=   0.9s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=200, subsample=0.9; total time=   1.3s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=500, subsample=0.5; total time=   2.3s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=500, subsample=0.7; total time=   3.2s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=500, subsample=1; total time=   4.4s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=100, subsample=0.9; total time=   1.3s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=100, subsample=1; total time=   1.6s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=200, subsample=0.5; total time=   1.7s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=200, subsample=0.9; total time=   2.3s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=500, subsample=0.5; total time=   3.6s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=500, subsample=0.7; total time=   4.8s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=500, subsample=1; total time=   6.2s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=100, subsample=0.7; total time=   3.1s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=100, subsample=0.9; total time=   3.7s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=200, subsample=0.5; total time=   5.2s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=200, subsample=0.9; total time=   7.6s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=200, subsample=1; total time=   6.3s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=500, subsample=0.7; total time=  13.0s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=500, subsample=0.9; total time=  15.0s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=100, subsample=0.5; total time=   7.3s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=100, subsample=0.7; total time=   9.6s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=100, subsample=1; total time=   4.5s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=200, subsample=0.5; total time=  14.0s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=200, subsample=0.7; total time=  19.8s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=200, subsample=1; total time=   8.7s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=500, subsample=0.5; total time=  34.0s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=500, subsample=0.9; total time=  57.9s
    [CV] END max_depth=None, max_features=sqrt, n_estimators=200; total time=   1.1s
    [CV] END max_depth=None, max_features=sqrt, n_estimators=300; total time=   1.6s
    [CV] END ...max_depth=None, max_features=2, n_estimators=300; total time=   1.2s
    [CV] END ...max_depth=None, max_features=4, n_estimators=200; total time=   0.9s
    [CV] END ...max_depth=None, max_features=8, n_estimators=100; total time=   0.6s
    [CV] END ...max_depth=None, max_features=8, n_estimators=200; total time=   1.2s
    [CV] END ..max_depth=None, max_features=16, n_estimators=100; total time=   0.9s
    [CV] END ..max_depth=None, max_features=16, n_estimators=100; total time=   0.9s
    [CV] END ..max_depth=None, max_features=16, n_estimators=300; total time=   3.1s
    [CV] END max_depth=None, max_features=log2, n_estimators=300; total time=   1.3s
    [CV] END max_depth=None, max_features=None, n_estimators=100; total time=   1.5s
    [CV] END max_depth=None, max_features=None, n_estimators=300; total time=   3.9s
    [CV] END ......max_depth=2, max_features=2, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=2, max_features=2, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=2, max_features=2, n_estimators=200; total time=   0.3s
    [CV] END ......max_depth=2, max_features=2, n_estimators=300; total time=   0.5s
    [CV] END ......max_depth=2, max_features=4, n_estimators=200; total time=   0.4s
    [CV] END ......max_depth=2, max_features=4, n_estimators=300; total time=   0.6s
    [CV] END ......max_depth=2, max_features=8, n_estimators=300; total time=   0.7s
    [CV] END .....max_depth=2, max_features=16, n_estimators=200; total time=   0.6s
    [CV] END .....max_depth=2, max_features=16, n_estimators=300; total time=   0.9s
    [CV] END ...max_depth=2, max_features=log2, n_estimators=300; total time=   0.6s
    [CV] END ...max_depth=2, max_features=None, n_estimators=200; total time=   0.8s
    [CV] END ...max_depth=3, max_features=sqrt, n_estimators=100; total time=   0.2s
    [CV] END ...max_depth=3, max_features=sqrt, n_estimators=200; total time=   0.5s
    [CV] END ...max_depth=3, max_features=sqrt, n_estimators=300; total time=   0.8s
    [CV] END ......max_depth=3, max_features=2, n_estimators=200; total time=   0.4s
    [CV] END ......max_depth=3, max_features=4, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=3, max_features=4, n_estimators=200; total time=   0.5s
    [CV] END ......max_depth=3, max_features=8, n_estimators=100; total time=   0.3s
    [CV] END ......max_depth=3, max_features=8, n_estimators=100; total time=   0.2s
    [CV] END ......max_depth=3, max_features=8, n_estimators=200; total time=   0.5s
    [CV] END .....max_depth=3, max_features=16, n_estimators=100; total time=   0.4s
    [CV] END .....max_depth=3, max_features=16, n_estimators=200; total time=   0.8s
    [CV] END .....max_depth=3, max_features=16, n_estimators=300; total time=   1.4s
    [CV] END ...max_depth=3, max_features=None, n_estimators=100; total time=   0.5s
    [CV] END ...max_depth=3, max_features=None, n_estimators=300; total time=   1.6s
    [CV] END ...max_depth=5, max_features=sqrt, n_estimators=200; total time=   0.6s
    [CV] END ...max_depth=5, max_features=sqrt, n_estimators=300; total time=   0.9s
    [CV] END ......max_depth=5, max_features=2, n_estimators=300; total time=   0.7s
    [CV] END ......max_depth=5, max_features=4, n_estimators=200; total time=   0.5s
    [CV] END ......max_depth=5, max_features=8, n_estimators=100; total time=   0.3s
    [CV] END ......max_depth=5, max_features=8, n_estimators=100; total time=   0.4s
    [CV] END ......max_depth=5, max_features=8, n_estimators=300; total time=   1.0s
    [CV] END .....max_depth=5, max_features=16, n_estimators=200; total time=   1.0s
    [CV] END .....max_depth=5, max_features=16, n_estimators=300; total time=   1.5s
    [CV] END ...max_depth=5, max_features=None, n_estimators=100; total time=   0.8s
    [CV] END ...max_depth=5, max_features=None, n_estimators=300; total time=   2.1s
    [CV] END ..max_depth=10, max_features=sqrt, n_estimators=200; total time=   0.8s
    [CV] END ..max_depth=10, max_features=sqrt, n_estimators=300; total time=   1.2s
    [CV] END .....max_depth=10, max_features=2, n_estimators=300; total time=   0.9s
    [CV] END .....max_depth=10, max_features=4, n_estimators=200; total time=   0.8s
    [CV] END .....max_depth=10, max_features=8, n_estimators=100; total time=   0.5s
    [CV] END .....max_depth=10, max_features=8, n_estimators=200; total time=   0.9s
    [CV] END .....max_depth=10, max_features=8, n_estimators=300; total time=   1.6s
    [CV] END ....max_depth=10, max_features=16, n_estimators=300; total time=   2.4s
    [CV] END ..max_depth=10, max_features=log2, n_estimators=200; total time=   0.8s
    [CV] END ..max_depth=10, max_features=log2, n_estimators=300; total time=   1.1s
    [CV] END ..max_depth=10, max_features=None, n_estimators=200; total time=   2.3s
    [CV] END ..max_depth=20, max_features=sqrt, n_estimators=100; total time=   0.5s
    [CV] END ..max_depth=20, max_features=sqrt, n_estimators=100; total time=   0.5s
    [CV] END ..max_depth=20, max_features=sqrt, n_estimators=200; total time=   1.0s
    [CV] END ..max_depth=20, max_features=sqrt, n_estimators=300; total time=   1.6s
    [CV] END .....max_depth=20, max_features=2, n_estimators=200; total time=   0.7s
    [CV] END .....max_depth=20, max_features=4, n_estimators=100; total time=   0.5s
    [CV] END .....max_depth=20, max_features=4, n_estimators=100; total time=   0.5s
    [CV] END .....max_depth=20, max_features=4, n_estimators=200; total time=   1.2s
    [CV] END .....max_depth=20, max_features=8, n_estimators=100; total time=   0.7s
    [CV] END .....max_depth=20, max_features=8, n_estimators=100; total time=   0.8s
    [CV] END .....max_depth=20, max_features=8, n_estimators=300; total time=   1.8s
    [CV] END ....max_depth=20, max_features=16, n_estimators=200; total time=   1.9s
    [CV] END ....max_depth=20, max_features=16, n_estimators=300; total time=   3.5s
    [CV] END ..max_depth=20, max_features=None, n_estimators=100; total time=   1.8s
    [CV] END ..max_depth=20, max_features=None, n_estimators=200; total time=   2.7s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=100, subsample=0.5; total time=   0.3s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=100, subsample=0.9; total time=   0.4s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=100, subsample=1; total time=   0.5s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=200, subsample=0.7; total time=   0.8s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=200, subsample=1; total time=   0.9s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=500, subsample=0.5; total time=   1.6s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=500, subsample=0.9; total time=   2.2s
    [CV] END learning_rate=0.01, max_depth=2, n_estimators=500, subsample=1; total time=   2.3s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=100, subsample=1; total time=   0.6s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=200, subsample=0.7; total time=   1.2s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=200, subsample=0.9; total time=   1.3s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=500, subsample=0.5; total time=   2.4s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=500, subsample=0.9; total time=   3.7s
    [CV] END learning_rate=0.01, max_depth=3, n_estimators=500, subsample=1; total time=   4.7s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=100, subsample=1; total time=   1.7s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=200, subsample=0.7; total time=   2.1s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=200, subsample=0.9; total time=   2.1s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=500, subsample=0.5; total time=   3.5s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=500, subsample=0.7; total time=   5.1s
    [CV] END learning_rate=0.01, max_depth=5, n_estimators=500, subsample=1; total time=   6.2s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=100, subsample=0.7; total time=   3.1s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=100, subsample=1; total time=   3.2s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=200, subsample=0.5; total time=   5.4s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=200, subsample=0.9; total time=   7.9s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=500, subsample=0.5; total time=  10.8s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=500, subsample=0.7; total time=  13.0s
    [CV] END learning_rate=0.01, max_depth=10, n_estimators=500, subsample=1; total time=  15.8s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=100, subsample=0.7; total time=   9.3s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=100, subsample=0.9; total time=  10.3s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=200, subsample=0.7; total time=  19.1s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=200, subsample=1; total time=   9.6s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=200, subsample=1; total time=   9.6s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=500, subsample=0.5; total time=  34.8s
    [CV] END learning_rate=0.01, max_depth=20, n_estimators=500, subsample=0.9; total time=  58.7s



```python
gbm_scores = cross_validate(gbm_final, X, y, cv=skfold, scoring=["accuracy","precision", "recall", "f1", "roc_auc"])
check_scoring(gbm_scores)
```

    Accuracy: 0.804
    Precision: 0.665
    Recall: 0.53
    F1: 0.59
    Roc_auc: 0.848


## 7. Feature Importance


```python
gbm_final_df = pd.DataFrame({"Value": gbm_final.feature_importances_, "Feature": X_train.columns}).sort_values('Value', ascending=False)
sns.set_style('white')
ax = sns.barplot(gbm_final_df, x='Value', y='Feature', palette='Reds_r')

for container in ax.containers:
    ax.bar_label(container, label_type='edge', padding=3, fmt="%.2f")

sns.despine(left=True, bottom=True)
ax.set_title('Feature Importance')
plt.show()
```


    
![png](output_87_0.png)
    


## 8. Explaining the Model


```python
explainer = shap.TreeExplainer(gbm_final)
shap_explanation = explainer(X_train)
shap_values = explainer.shap_values(X_train)
```


```python
shap.summary_plot(shap_values, X_train)
```

    No data for colormapping provided via 'c'. Parameters 'vmin', 'vmax' will be ignored



    
![png](output_90_1.png)
    



```python
shap.dependence_plot("tenure", shap_values, X_train)
```


    
![png](output_91_0.png)
    



```python
shap.dependence_plot("MonthlyCharges", shap_values, X_train, interaction_index=None)
```


    
![png](output_92_0.png)
    



```python
shap.dependence_plot("TotalCharges", shap_values, X_train)
```


    
![png](output_93_0.png)
    


## 9. Test Prediction


```python
predictions = gbm_final.predict(X_test)
```


```python
output = pd.DataFrame({"Customer": X_test.reset_index()['index'], "Churn":predictions})
output.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>185</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2715</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3825</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1807</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>132</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 10. Conclusion

Final model: Gradient Boosting

Scorings can be further improved by testing several alternative algorithm.

Recommendations to business stakeholders:

1. Conduct user surveys to understand the factors causing user churn.
2. Conduct time series analysis to understand if there was a particular shift in churn rate at a certain point in time. If there was, identifying that particular time could help to isolate and further understand churn reasons. For example, perhaps there was a price change or product change that inadvertedly increased churn.
3. Acquiring new customers are always more expensive than retaining existing customers. It's important to devise retention scheme or retention bonus to encourage users to continue using the service.
4. Conduct user surveys on EXISTING users who are predicted to churn. Key to understand current users' frustrations and solve for it before they decide to churn.
5. Conduct a user test for fiber optic internet service to see if there's any major issue with the service. There could potentially be an SLA issue or mismatch of users' expectation with what's provided in the service.
6. Similar to point no. 5, a check should also be conducted on the electronic check payment method to understand if there's any issue a user might encounter when using that as a payment method.
