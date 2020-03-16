## [Overview](../index.md)

## [Read and Assess](.../read_assess/read_assess.md)

## [Preprocessing](.../preprocessing/cleaning.md)

## [Analyze and Visualize](.../analyze_visualize/analyze_visualize.md)

## [Model Building](.../model_building/model.md)

# Conclusion

### Summary
In this analysis, I tried to understand the relationship between features and price.By doing data analysis and visualization, We looked into some features affects on price seperately. Then the Random Forest model further helped me validate my analysis result and pick import features. 

We could know that the basic characteristics of place, the host quality and time of year would affect price. If you are host, this will help you make decision on your price. If you want to book a place, this will help you find the favoriate and the cheapiest place.


### Model Improvements

Since the initial baseline exploration and modeling presented on the poster below, much more cleaning, feature building, model tuning and more helped improve the model's pricing prediction accuracy. Namely some of these enhancements included:

* Trimming categorical features to keep sample size down after one-hot encoding.
* Feature construction based on distances like *proximity to station* and *on central park*.
* Feature construction from sentiment analysis (TextBlob Package gave best results).
* Imputation using K-Nearest Neighbors on all features with missing data in our final dataset.
* Normalize response and explantory variable to fix the right skew data

