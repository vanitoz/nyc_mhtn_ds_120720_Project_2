import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta

from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
from geopy.distance import distance
from statsmodels.formula.api import ols
import statsmodels.api as sm


import scipy.stats as scs
from scipy.stats import norm
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression,mutual_info_regression
from sklearn.feature_selection import RFECV
import pickle


from sklearn.preprocessing import StandardScaler
from sklearn import linear_model


def train_test_valid(df_features, target):
    """
    Function to create Train and test split
    Fit linear regression to the data 
    Evaluate the model 
    
    params : df_features - Data frame with features
             target - series with numeric values !!!! target should be np.log(target)
    """
    #call train_test_split on the data and capture the results # randomles is consistent
    X_train, X_test, y_train, y_test = train_test_split(df_features, target, random_state=9,test_size=0.2)
    
    #instantiate a linear regression object
    lm = linear_model.LinearRegression()
    
    #fit the linear regression to the data
    lm = lm.fit(X_train, y_train)
    
    y_train_pred = lm.predict(X_train)
    
    # we have the predictions, we need to exponentiate them to get them back into the original scale, dollars
    #y_train_pred = np.exp(y_train_pred)
    
    train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
    #print('Root Mean Squared Error:' , train_rmse)
    
    #Predicting the Test Set
    y_pred = lm.predict(X_test)
    #y_pred = np.exp(y_pred)
    
    test_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    #print('Root Mean Squared Error:' + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
    
    #Comparing our Model's performance on training data versus test data
    print('Training: RMSE', int(train_rmse), "vs. Testing: RMSE", int(test_rmse))
    print('Perfomance : {} %'.format(round(abs((test_rmse-train_rmse)/train_rmse)*100)))
    
    
    
    
def features_corr_matrix(df, threshold):
    
    """
    Plot corr matrix of btw df columns
    
    params: 
            df - dataframe
            threshold - critical value for feature selection
    return: 
            plot, list of sugested features to exclude
    """
    # Create correlation matrix    
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    sns.set(style="white")
    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
   
    plt.show()
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    return  print('Based on threshold {} , sugested featrures to drop - {}'.format(threshold,to_drop))
 
    
    
def box_plots(df):
    """
    Plot boxplots 
    params: 
            df - with only continuous, numeric values
    """
    fig = plt.figure(figsize=(25,20))
    for f,n in zip(df.columns, range(1,len(df.columns)+1)):
        plt.subplot(int(len(df.columns)/2+0.5),2,n)
        plt.title("{}".format(f))
        sns.boxplot(data = df[f], orient='h', color = 'b')
    return plt.show()


def relationship(df, y):
    """
    Build plot to show relationship btween df values and y values
    params: 
            df - continuous, numeric values only
            y - continuous, numeric values only (y-axis)
    """
    fig = plt.figure(figsize=(25,20))
    for f,n in zip(df.columns, range(1,len(df.columns)+1)):
        #plt.subplot(int(len(df.columns)/2+0.5),2,n)
        #plt.title("{}".format(f))
        sns.jointplot(x=df[f], y = y, kind='reg')
    return plt.show()


def map_bed_bath(row):
    """
    Checking extreame number of rooms in the house
    If there is 0 rooms - replace with number of floors in the house
    If there is more then 10 rooms - replace with first digit 
    Of the number of rooms in the house
    """
    
    if row['bedrooms'] == 0:
        row['bedrooms'] = row['floors']
    
    if row['bathrooms'] == 0:
        row['bathrooms'] = row['floors']
    
    if row['bedrooms'] > 10 :
        row['bedrooms'] = int(str(row['bedrooms'])[0])
        
    return row

def distance_to_dwntwn(row):    
    dntown = (47.608013, -122.335167)
    coord = (row['lat'],row['long'])
    dist = distance(dntown, coord).miles
    return int(dist)


def to_dummies(df, features):
    """
    Generating dummy variables for features
    parama: df
            features - list of features
    """
    d_df = []
    for f in features:
        d_df.append(pd.get_dummies(df[f], prefix='{}'.format(str(f)[:3]), drop_first=True))
    #import pdb;pdb.set_trace()
    df = df.drop(features, axis = 1)
    df = pd.concat([df] + d_df ,axis=1)
    
    return df

def format_yr_renov(row):
    """
    Convert raw into datetime format

    """
    if row['yr_renovated'] == 0:
        row['yr_renovated'] = datetime.datetime.now()
    else: 
        row['yr_renovated'] = pd.to_datetime(row['yr_renovated'], format='%Y')
    return row


def create_poly_df (df, degree):
    
    """
    Generate polinomial features with degree
    Returns DF with poly futers
    params:
            df - dataframe with features. numeric type
            degree - int
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_data = poly.fit_transform(df)
    
    poly_columns = poly.get_feature_names(df.columns)
    df_poly = pd.DataFrame(poly_data, columns=poly_columns)
    
    return df_poly


def scale_transform_validate(df, target, residuals = False, selection = False):
    
    """
    Scale features and Evaluate model
    params : df_features - Data frame with features (numeric)
             target - series with numeric values!  
             residuals = False. Plot residuals if True 
    """
    
    X_train, X_test, y_train, y_test = train_test_split(df, target, random_state=9, test_size=0.2)
    scaler = StandardScaler()
    
    # fit the scaler to the training data
    scaler.fit(X_train)
    
    #transform the training data
    scaled_data = scaler.transform(X_train)
    
    # create DF
    X_train_scaled = pd.DataFrame(data=scaled_data, columns=df.columns, index=X_train.index)
    
    #transform the test data
    scaled_test_data = scaler.transform(X_test)
    
    #create dataframe
    X_test_scaled = pd.DataFrame(data=scaled_test_data, columns=df.columns, index=X_test.index)
    
    # Fit the model to the training data.
    lm = LinearRegression()
    lm = lm.fit(X_train_scaled, y_train)
    
    # Use the model to predict on the training set and the test set.
    y_train_pred = lm.predict(X_train_scaled)
    y_test_pred = lm.predict(X_test_scaled)
    
    train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    
    if residuals == True:
        sns.residplot( y_test, y_test_pred, lowess =True, color='g')
    
    if selection == True:
        
        return round(train_rmse), round(test_rmse)
    
    #Comparing our Model's performance on training data versus test data
    else:
        print('Training: RMSE', int(train_rmse), "vs. Testing: RMSE", int(test_rmse))
        print('Perfomance : {} %'.format(round(abs((test_rmse-train_rmse)/train_rmse)*100)))

    return plt.show()
    


def f_test_selection(df, target, k):
    """
    k - Select features according to the k highest scores
    """
    X_train, X_test, y_train, y_test = train_test_split(df, target, random_state=9, test_size=0.2)
    scaler = StandardScaler()

    # fit the scaler to the training data
    scaler.fit(X_train)

    #transform the training data
    scaled_data = scaler.transform(X_train)

    X_train_scaled = pd.DataFrame(data=scaled_data, columns=df.columns, index=X_train.index)

    #transform the test data
    scaled_test_data = scaler.transform(X_test)

    #create dataframe
    X_test_scaled = pd.DataFrame(data=scaled_test_data, columns=df.columns, index=X_test.index)

    selector = SelectKBest(f_regression, k= k)
    selector.fit(X_train_scaled, y_train)
    
    selected_columns = X_train_scaled.columns[selector.get_support()]
    removed_columns = X_train_scaled.columns[~selector.get_support()]
    
    return selected_columns


def f_test_select_vis(df, target):
    
    trn =[]
    tst =[]
    k = []
    for f in range(1,len(df.columns)):

        best_f = f_test_selection(df, target = target, k = f)
        train_r,test_r = scale_transform_validate(df[best_f], target = target, selection = True)

        trn.append(train_r)
        tst.append(test_r)
        k.append(f)
    
    df = pd.DataFrame(data = list(zip(trn,tst,k)), columns =['RMSE Training','RMSE Testing','Number of features']) 
    
    sns.lineplot(x='Number of features', y='value', hue='variable', 
             data=pd.melt(df, ['Number of features']))
    
    return plt.show()

def scale_fit_pickle_origin(df_features, target):
    
    """
    Scaling df with features,
    Fit linear model with scaled features
    Create pickle file with Scaler and Model
    
    params: 
            df_features - most important features 
            target - Series
    """
    
    scaler = StandardScaler()

    #fit the scaler to the training data
    scaler.fit(df_features)

    #transform the training data
    scaled_data = scaler.transform(df_features)
    
    #create dataframe
    df_features_scaled = pd.DataFrame(data=scaled_data, columns=df_features.columns, index=df_features.index)
    
    
    lm_final = LinearRegression()

    #fit the linear regression to the data
    lm_final = lm_final.fit(df_features, target)
    
    pickle_out = open("model.pickle","wb")
    pickle.dump(lm_final, pickle_out)
    pickle_out.close()

    pickle_out = open('scaler.pickle', "wb")
    pickle.dump(scaler, pickle_out)
    pickle_out.close()
    
    print(' CONGRATS !!! You sucessfuly created you pickles for SCALER and MODEL')
    
    return lm_final



def diff_two_means(sample1, sample2, threshold):
    
    x1_m = sample1['price'].mean()
    x2_m = sample2['price'].mean()
    
    x1_s = sample1['price'].std()
    x2_s = sample2['price'].std()
    
    n1 = sample1['price'].count()
    n2 = sample2['price'].count()
    
    alpha = (1-(threshold)/100)/2
    df = len(sample1)+len(sample2) - 1
    t_crit = scs.t.ppf(1 - alpha, df=df)   # critical value 
    numer = (x1_m- x2_m) - 0
    denum = math.sqrt((x1_s**2/n1)+(x2_s**2/n2))
    delta_mu = numer/denum
    
    
    p = 1 - scs.t.cdf(delta_mu, df=df) 
    
    if delta_mu > t_crit or delta_mu < - t_crit :
        
        return print("""
        We reject the null Hypotesys because based one statistical test two groups sample means difference 
        is = {}, which gets in rejection area defined by critical values {} and -{}.
        """.format(round(delta_mu,2),round(t_crit,2),round(t_crit,2)))
    
    else:
        return print("""
        There is not enough evidence to reject the null Hypotesys because based one statistical test
        two groups sample means difference is = {}, which does not get in rejection area defined by critical values {} and -{}.
        """.format(round(delta_mu,2),round(t_crit,2),round(t_crit,2)))
    
    
    
    def anova_test(df, variable, threshold):
        """
        params : df - dataframe 
                 variable - String / collumn name in data
                 trashold - integer as pct. confidance. level
        """
        alpha = round(1-threshold/100,2)
        # t_crit = scs.t.ppf(alpha, df=len(df)-1)
        # fit the model / can yopu predict a phys_halth based on state
        anova_states = ols('{}~condition'.format(variable), data=df).fit()
        anova_table = sm.stats.anova_lm(anova_states, type=2)
        # print(anova_table)
        f_score = round(anova_table['F']['condition'],2)
        # lets check what is the probability to get f_score or biger
        pr = anova_table['PR(>F)']['condition'] 

        if pr < alpha:

            return print( """
            We reject the null Hypotesys because the test statistic falls in the rejection area.
            Based one statistical test our value pr = {}, which smaller then trashold = {}.
            """.format(pr,alpha))

        else:

            return print("""
            There is not enough evidence to reject the null Hypotesys because the test statistic does not fall in the rejection area.
            Based one statistical test our value pr = {}, which way smaller then trashold = {}.
            Another way we couldn't conclude that there is relationship between state and phys_health of patients. 
                        """.format(pr,alpha))
