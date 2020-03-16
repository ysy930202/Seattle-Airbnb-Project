## [Overview](../index.md)

# Read and Assess

## [Preprocessing](.../preprocessing/cleaning.md)

## [Analyze and Visualize](.../analyze_visualize/analyze_visualize.md)

## [Model Building](.../model_building/model.md)

## [Conclusion](.../conclusion/conclusion.md)



### Import Libraries


```python
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
pd.set_option("max_columns",None)

# Import plotting packages
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


#Import machine learning
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split #split
from sklearn.metrics import r2_score, mean_squared_error #metrics
```


```python
# Global variables
BNB_BLUE = '#007A87'
BNB_RED = '#FF5A5F'
BNB_DARK_GRAY = '#565A5C'
BNB_LIGHT_GRAY = '#CED1CC'
```

### Read Dataset - listings & Calendar

```python
# Read in the calendar date
calendar = pd.read_csv('C:/Udacity Data Scientist/Project 1 - Airbnb Data/seattle/calendar.csv')
calendar.head()
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>listing_id</th>
      <th>date</th>
      <th>available</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>241032</td>
      <td>2016-01-04</td>
      <td>t</td>
      <td>$85.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>241032</td>
      <td>2016-01-05</td>
      <td>t</td>
      <td>$85.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>241032</td>
      <td>2016-01-06</td>
      <td>f</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
listings = pd.read_csv('C:/yoyo/course/Udacity Data Scientist/Project 1 - Airbnb Data/seattle/listings.csv')
listings.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>scrape_id</th>
      <th>...</th>
      <th>host_listing_count</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>241032</td>
      <td>20160104002432</td>
      <td>...</td>
      <td>3</td>
      <td>$85.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>953595</td>
      <td>20160104002432</td>
      <td>...</td>
      <td>6</td>
      <td>$150.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3308979</td>
      <td>20160104002432</td>
      <td>...</td>
      <td>2</td>
      <td>$975.00</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 52 columns</p>
</div>


### Assess data - NaN in the listings and Calendar


```python
def plot_na(df,figsize):
    
    '''
    INPUT:
    df - DataFrame
    figsize - figure size
    
    OUTPUT:
    df_na - columns with missing value
    plot - percent of missing value
    
    '''
    
    df_na = df.isna().mean()[df.isna().mean() > 0] * 100
    df_na = df_na.sort_values(ascending = False)
#     print('Columns with NaN List')
    df_na = pd.DataFrame(df_na, columns = ['Percent of NaN'])
    print(df_na)
    
    # plot
    ax = df_na.plot(kind = 'bar', figsize = figsize, color = BNB_BLUE, alpha = 0.85)
    ax.set_xlabel('Columns with missing value')
    ax.set_ylabel('Percent of NaN %')
    ax.set_title('Missing values per column, %')
    
    return plt.show()
```


```python
# NA of Calendar 
plot_na(calendar,(6,4))
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NaN Columns</th>
      <th>% of NaN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>price</td>
      <td>33.0</td>
    </tr>
  </tbody>
</table>
</div>


![png](output_10_0.png)


```python
# NA of listings data
plot_na(listings, (12,8))
```


![png](output_14_0.png)


