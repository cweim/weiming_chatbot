# Natural Language Processing for Stock Market Indicator

Timeline: January 22, 2024 → April 29, 2024
Client: SUTD
My Role: Data Analyst, Data Science, Machine Learning
Tools: Hugging Face Transformers, Pandas, Python
Document: ../team9.pdf

## Overview

**Objective:** Develop a reliable stock market indicator based on Twitter sentiment for stocks like AAPL, AMZN, GOOG, MSFT, and TSLA.

![photo_2024-08-27 17.24.41.jpeg](Natural%20Language%20Processing%20for%20Stock%20Market%20Indic%20092c857fce644727b25f0e5bdc450e4a/photo_2024-08-27_17.24.41.jpeg)

**Approach:**

- **Data Collection:** Assembled a multivariable dataset combining historical stock market data with aggregated Twitter sentiment with pandas library.
    
    ![Screenshot 2024-08-27 at 5.26.55 PM.png](Natural%20Language%20Processing%20for%20Stock%20Market%20Indic%20092c857fce644727b25f0e5bdc450e4a/Screenshot_2024-08-27_at_5.26.55_PM.png)
    
    ![Screenshot 2024-08-27 at 5.31.15 PM.png](Natural%20Language%20Processing%20for%20Stock%20Market%20Indic%20092c857fce644727b25f0e5bdc450e4a/Screenshot_2024-08-27_at_5.31.15_PM.png)
    
- **Financial Sentiment Analysis:** Utilised the Hugging Face FinBERT model to analyse daily sentiment of tweets. Calculated a weighted average sentiment score, factoring in likes, retweets, and comments.
- **Modelling:** Explored the Random Forest Classifier, performing 5-fold cross-validation and Grid Search CV to optimised parameters.
    
    ![Screenshot 2024-08-27 at 5.35.12 PM.png](Natural%20Language%20Processing%20for%20Stock%20Market%20Indic%20092c857fce644727b25f0e5bdc450e4a/Screenshot_2024-08-27_at_5.35.12_PM.png)
    

**Outcome:**

- Achieved 85.2% accuracy in predicting daily stock market indicators.
    
    ![Screenshot 2024-08-27 at 5.33.52 PM.png](Natural%20Language%20Processing%20for%20Stock%20Market%20Indic%20092c857fce644727b25f0e5bdc450e4a/Screenshot_2024-08-27_at_5.33.52_PM.png)
    
- Demonstrated that Twitter sentiment significantly influences market movements.
    
    ![Screenshot 2024-08-27 at 5.33.21 PM.png](Natural%20Language%20Processing%20for%20Stock%20Market%20Indic%20092c857fce644727b25f0e5bdc450e4a/Screenshot_2024-08-27_at_5.33.21_PM.png)