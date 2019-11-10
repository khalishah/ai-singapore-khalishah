#!/usr/bin/env python
# coding: utf-8

def data_prep():
    import pandas as pd, sys, re, datetime, time
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    from datetime import datetime, timedelta

    get_ipython().run_line_magic('matplotlib', 'inline')
    
    # Read data
    url = "https://aisgaiap.blob.core.windows.net/aiap5-assessment-data/traffic_data.csv"
    df = pd.read_csv(url, parse_dates = ['date_time'])
    df.drop('snow_1h', axis = 1, inplace = True)
    
    # Fix weather_description
    df['weather_description'] = df['weather_description'].str.lower()
    
    # Get the repeated date_time values
    rep_date = pd.DataFrame(df['date_time'].value_counts()).reset_index()
    rep_date = pd.DataFrame(rep_date[rep_date['date_time'] > 1]['index'])
    rep_date.rename(columns = {'index': 'date_time'}, inplace = True)
    
    # To ensure concatenation of weather_main strings is done correctly, avoid duplicates
    df.sort_values(by = 'weather_main', inplace = True)
    df.reset_index(inplace = True, drop = True)
    
    # Concatenation of weather_main and weather_description
    df_rep = pd.merge(df, rep_date, how = 'inner', on = 'date_time')
    df_rep_wt_main = pd.DataFrame(df_rep.groupby(df_rep.columns.difference(['weather_main', 'weather_description']).tolist())['weather_main'].apply(lambda x: "%s" % ', '.join(x.unique()))).reset_index()
    df_rep_wt_desc = pd.DataFrame(df_rep.groupby('date_time')['weather_description'].apply(lambda x: "%s" % ', '.join(x.unique()))).reset_index()
    df_rep = pd.merge(df_rep_wt_main, df_rep_wt_desc, how = 'inner', on = 'date_time')
    
    # Remove the repeated rows of date_time
    df.drop_duplicates(subset =['date_time', 'traffic_volume'], keep = False, inplace = True)
    
    # Append the new cleaned rows of date_time
    df = df.append(df_rep, sort = True)
    
    # Extract date features from date_time
    def add_datepart(df, date_field, drop=False, time=False, errors="raise"):
        fld = df[date_field]
        fld_dtype = fld.dtype
        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            fld_dtype = np.datetime64
    
        if not np.issubdtype(fld_dtype, np.datetime64):
            df[date_field] = fld = pd.to_datetime(fld, infer_datetime_format=True, errors=errors)
        targ_pre = re.sub('[Dd]ate$', '', date_field)
        attr = ['Month', 'Day', 'Hour', 'Date','Weekday']
        if time: 
            attr = attr + ['Hour', 'Minute', 'Second']
        for n in attr: 
            df[targ_pre + n] = getattr(fld.dt, n.lower())
        if drop: 
            df.drop(date_field, axis=1, inplace=True)
        
    add_datepart(df, 'date_time')
    
    # Fix Date data type
    df['date_timeDate'] =  pd.to_datetime(df['date_timeDate'], format='%Y-%m-%d')
    
    # Clean Holiday feature
    df_hols = df[df['holiday'] != 'None'][['holiday', 'date_timeDate']]

    df = pd.merge(df, df_hols, how = 'left', on = ['date_timeDate']).drop(columns = 'holiday_x')
    df.rename(columns = {'holiday_y': 'holiday'}, inplace = True)
    df['holiday'].fillna('None', inplace = True)

    df['is_Holiday'] = df['holiday'].apply(lambda x: 1 if x != 'None' else 0)

    df_holiday = df[df['is_Holiday'] == 1][['holiday', 'date_timeDate']].drop_duplicates()

    post = pd.DataFrame(df_holiday['date_timeDate'] + timedelta(days = 1))
    post['prepost_holiday'] = 'Post '+ df_holiday['holiday']
    post['is_prepost_hols'] = 2
    pre = pd.DataFrame(df_holiday['date_timeDate'] + timedelta(days = -1))
    pre['prepost_holiday'] = 'Pre '+ df_holiday['holiday']
    pre['is_prepost_hols'] = 1
    hols = pre.append(post, ignore_index = True)

    df = pd.merge(df, hols, how = 'left', on = ['date_timeDate'])
    df['prepost_holiday'].fillna('None', inplace = True)
    df['is_prepost_hols'].fillna(0, inplace = True)

    # Prepare Categorical Features - to be encoded

    df_cat = df[['date_time', 'weather_main', 'holiday']]
    df_cat_encoded = pd.get_dummies(df_cat)
    df = pd.merge(df, df_cat_encoded, how = 'inner', on = ['date_time'])

    # Drop Unecessary Features for final dataframe

    df.drop(columns = ['weather_main', 'weather_description', 'holiday', 'prepost_holiday', 'date_time', 'date_timeDate'], inplace = True)
    
    return df


def traffic_model(df):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn import metrics
    import math
    
    # Get independent and dependent variable
    X = df.drop('traffic_volume', axis=1)
    y = df['traffic_volume']
    
    # Train-test_validation Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=10)
    
    # Metrics used to validate: RMSE
    def rmse(x,y): return math.sqrt(((x-y)**2).mean())
    
    # To print scores
    def print_score(m):
        res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                    m.score(X_train, y_train), m.score(X_valid, y_valid)]
        if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
        print('\nRoot Mean Squared Error of Predicted and Actual in on Training Set: '+ str(res[0]))
        print('Root Mean Squared Error of Predicted and Actual in Validation Set: '+ str(res[1]))
        print('R Squared score using Training Set: '+ str(res[2]))
        print('R Squared score using Validation Set: '+ str(res[3]))
    
    # Final model
    model = RandomForestRegressor(n_estimators=40, n_jobs=-1)
    model.fit(X_train, y_train)
    return print_score(model)


def main():
    df = data_prep()
    return traffic_model(df)


main()




