
# King County Housing Price Prediction
![seattle](images/seattle_img.jpg)

## Overview

The pandemic has certainly affected every sector but residential real estate has been very resilient. The real estate sector has also been highly supportive of the economic recovery of the country so far. It has emerged as a pillar of support for the economy. 2020 was a record-breaking year for the US housing market. According to Zillow, in total, 5.64 million homes were sold in 2020, up 5.6% from 2019. Buyers have to face more competition and act more quickly than usual to snag their dream home. That's how hot the real estate market has been throughout the pandemic. Thankful for technologies, nowadays, we are able to create the models that will predict the price of a house given certain features such as geolocation, squre feet of living, size of basement, school rate of region etc.

This project will utilize multiple regression analysis to predict housing prices in King County, Seattle, WA. The training data will be explored for feature building and the final model will be built and trained by SciKit python library. The data and project scope was provide by Flatiron School for Data Science Immersive program phase 2 final project. 

## Repository Structure
    .
    ├── main.ipynb                          # main program notebook, model implementation
    ├── eda.ipynd	                        # project notebook with EDA and model creation
    ├── housing_preds_ivan.csv              # project resukts (predictions for kc_house_data_test_features.csv)
    ├── images                              # project image/graph files
    ├── scaler.pickle                       # scaler to the data
    ├── model.pickle                        # final regression models pickle
    ├── kc_house_data_test_features.csv     # test set of data for prediction
    ├── kc_house_data_train.csv             # train set of data to create a model
    └── README.md

## Business Problem
Predict housing prices using contextual data in King County, specifically around Seattle, WA. The aim is to generate accurate sale predictions and will aid in strategizing the investment options to maximize profit for real estate companies that are interested in use this model. The multiple linear regression model will be used as the basis with provided property data for this task.

## Approach

1. Check for data completeness and integrity
2. Perform EDA with statistical analysis to determine statistically significant features
3. Visualize statistically significant features
4. Engineer new features based on stastistical findings
5. Model Linear Regression models and evaluate each model for final implementation
6. Implement feature engineering and final model to the data set. 

## Analisis
The location of the property was one of the most important factor when determining the property value. It was found that the properties located closer to downtown got hihjer average price.<br>

![dist_dntwn](images/dist_dntwn.png)<br>

Analising more data we discover that having a renovated property significantly raised the average property price. Basement was a significant feature as well, but not as much as renovation.


![bsmn_renow](images/basement_renov.png)<br>

Properties with condition 4 and 5 had lower average price than properties with conditions 3 and 5.

![Condition](images/cond_score.png)<br>



# Kings County Housing Bake-off

For many machine learning projects, the goal is to create a model that best predicts the target variable on unseen data. In order to develop a model, we have a general process, but there is a lot of flexibility within this process. Even when working with the same data, people can produce different models by engineering different features, or by selecting certain features to include in the models. **There is no one correct way to create a model**.

For Phase 2, you will be creating a model that will **predict the prices of homes** sold in the Seattle, WA area. For this project there will be **3 deliverables**:

- a Github repo for this project
- a notebook showing your final modeling process
- a CSV file of your predictions on the holdout set
	- name this file `housing_preds_your_name.csv` (replacing `your_name` with your name) and send via Slack

## Holdout predictions

You will develop a model using `kc_house_data_train.csv`. Then you will use that model/process to predict on the `kc_house_data_holdout_features.csv`. 

***Important note #1***: If you create a new feature with your training data, you will need to do the same thing with the test data before using the model to predict on the holdout data.  

After using your model to predict the holdout data, you will submit those predictions as a `.csv` file to the instructional staff. We will score the submitted predictions using the RMSE of those predictions.

***Important note #2***: While we will score and rank each submission, your class rank will **not** have any direct impact on passing Phase 2. *The goal is to make sure you can actually produce predictions*.

So as long as you successfully **complete the modeling process** and can **explain the work you did**, you will be able to pass.  

## Final notebook

Through the modeling process, you will try many different techniques (**feature engineering** and **feature selection**, for example) to try and create your best model. Some will work and some will not lead to a better model. Through your modeling process, you will identify what actions create the best model. After you have finalized your process, you must create a 'cleaned up' and annotated notebook that shows your process.

Your notebook must include the following:

- **Exploratory Data Analysis (EDA):** You must create **at least 4 data visualizations** that help to explain the data. These visualizations should help someone unfamiliar with the data understand the target variable and the features that help explain that target variable.

- **Feature Engineering:** You must create **at least 3 new features** to test in your model. Those features do not have to make it into your final model, as they might be removed during the feature selection process. That is expected, but you still need to explain the features you engineer and your thought process behind why you thought they would explain the selling price of the house.  

- **Statistical Tests:** Your notebook must show **at least 3 statistical tests** that you preformed on your data set. Think of these as being part of your EDA process; for example, if you think houses with a view cost more than those without a view, then perform a two-sample T-test. These can be preliminary evidence that a feature will be important in your model.  

- **Feature Selection:** There are many ways to do feature selection: filter methods, P-values, or recursive feature elimination (RFE). You should try multiple different techniques and combinations of them. For your final model, you will **settle on a process of feature selection**; this process should be **clearly shown in your final notebook**.

- **Model Interpretation:** One of the benefits of a linear regression model is that you can **interpret the coefficients** of the model **to derive insights**. For example, which feature has the biggest impact on the price of the house? Was there a feature that you thought would be significant but was not? Think if you were a real estate agent helping clients price their house: what information would you find most helpful from this model?

## GitHub Repository

A GitHub repo is a good way to keep track of your work, but also to display the work you did to future employers. Your GitHub should contain the following:

- A `README.md` that briefly describes the project and the files within the repo.
- Your cleaned and annotated notebook showing your work.
- A folder with all of your 'working' notebooks where you played around with your data and the modeling process.

## Data Set Information

This data set contains information about houses that were sold in the Seattle area during the last decade. Below is a description of the column names, to help you understand what the data represents. As with most real world data sets, the column names are not perfectly described, so you'll have to do some research or use your best judgment if you have questions relating to what the data means. 

Like every data set, there are some irregularities and quirks. Trust me, there wasn't a house sold with 33 bedrooms, even though the data says there was. *You have to decide how you want to handle that example*. Also, some houses were sold more than once within the time frame of this dataset. Think about how that can be useful information in predicting the selling price.

As you go through this modeling process, think about what determines how much someone will pay for a house.  For example, the larger the house is, the more people will pay for it. If you understand why certain houses cost more than others and represent that in your model, it will be a more accurate model.  

Have fun!

# Column Names and descriptions for Kings County Data Set
* **id** - unique ID for a house
* **date** - Date day house was sold
* **price** - Price is prediction target
* **bedrooms** - Number of bedrooms
* **bathrooms** - Number of bathrooms
* **sqft_living** - square footage of the home
* **sqft_lot** - square footage of the lot
* **floors** - Total floors (levels) in house
* **waterfront** - Whether house has a view to a waterfront
* **view** - Number of times house has been viewed
* **condition** - How good the condition is (overall)
* **grade** - overall grade given to the housing unit, based on King County grading system
* **sqft_above** - square footage of house (apart from basement)
* **sqft_basement** - square footage of the basement
* **yr_built** - Year when house was built
* **yr_renovated** - Year when house was renovated
* **zipcode** - zip code in which house is located
* **lat** - Latitude coordinate
* **long** - Longitude coordinate
* **sqft_living15** - The square footage of interior housing living space for the nearest 15 neighbors
* **sqft_lot15** - The square footage of the land lots of the nearest 15 neighbors
