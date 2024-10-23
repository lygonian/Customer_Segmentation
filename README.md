# E_commerce Customers Segmentation

---

This notebook provides a comprehensive analysis and prediction model for a bank marketing campaign dataset. The key steps include data preparation, feature engineering, exploratory data analysis (EDA), RFM segmentation, and machine learning model training and evaluation. The final model, a `BalancedRandomForestClassifier`, achieved a significant improvement in predicting customer subscriptions to term deposits, with a notable F2-score of 0.70 (increased from 0.30). (Notebook below)

Here a summarizing dash app which lets you filter between Customer Segments.
<iframe src="https://customer-segmentation-cbdg.onrender.com/" width="100%" height="600"></iframe>  

---  

# Notebook
View notebook [here](https://nbviewer.org/github/lygonian/Customer_Segmentation/blob/master/notebooks/customer_segementation.ipynb)  
File saved [here](https://github.com/lygonian/Customer_Segmentation/blob/master/notebooks/customer_segementation.ipynb)  
The analysis includes data preparation, feature engineering, client profiling, RFM segmentation, and prediction using machine learning models. The final model achieved a significant improvement in predicting customer subscriptions to term deposits.  
---  
  
# Goal
Term deposits are a major source of income for a bank. Your money is invested for an agreed rate of interest over a fixed amount of time, or term. The bank has various outreach plans to sell term deposits to their customers such as email marketing, advertisements, telephonic marketing, and digital marketing.  

Telephonic marketing campaigns still remain one of the most effective way to reach out to people. However, they require huge investment as large call centers are hired to actually execute these campaigns. Hence, it is crucial to identify the customers most likely to convert beforehand so that they can be specifically targeted via call.

The data is related to direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe to a term deposit (variable y).

---

# Methods
1. **Data Preparation and EDA**:
    - The dataset was thoroughly cleaned and prepared, including handling missing values and duplicates.
    - Feature engineering was performed to create meaningful attributes such as age groups, job categories, and RFM scores.

2. **Client Profiling**:
    - Detailed profiling of clients based on personal information, banking details, and campaign interactions was conducted.
    - Key insights were derived, such as the characteristics of clients who are more likely to subscribe to term deposits.

3. **RFM Segmentation**:
    - Customers were segmented based on Recency, Frequency, and Monetary values.
    - Segments such as Champions, Potential Loyalists, and At-Risk Customers were identified, providing valuable insights for targeted marketing strategies.

4. **Prediction Model**:
    - A `BalancedRandomForestClassifier` was trained and evaluated, achieving a high F2-score.
    - Feature importance analysis helped in selecting the most relevant features for the model.


# Future Implications
1. **Enhanced Marketing Strategies**:
    - The insights from RFM segmentation and client profiling can be used to develop more targeted and effective marketing campaigns.
    - Personalized marketing strategies can be designed for different customer segments to improve engagement and conversion rates.

2. **Model Improvement**:
    - Further hyperparameter tuning and experimentation with different machine learning algorithms can be conducted to improve model performance.
    - Incorporating additional data sources, such as customer transaction history or social media interactions, could enhance the predictive power of the model.

3. **Real-time Prediction**:
    - The model can be deployed in a real-time environment to provide instant predictions and recommendations for marketing campaigns.
    - Integration with customer relationship management (CRM) systems can automate the process of identifying potential customers for term deposits.

4. **Continuous Monitoring and Updating**:
    - Regular monitoring and updating of the model with new data will ensure its accuracy and relevance over time.
    - Implementing a feedback loop to capture the outcomes of marketing campaigns can help in refining the model and strategies continuously.


# Summary of Feature Insights

### 1. Age
- **Insight**: Age is a significant factor in predicting deposit subscriptions. Younger adults (18-30) and seniors (65+) are more likely to subscribe to term deposits.
- **Actionable Strategy**: Tailor marketing campaigns to target these age groups with specific products and services that cater to their needs.

### 2. Job
- **Insight**: Job categories such as management, retired, and students show higher subscription rates for term deposits.
- **Actionable Strategy**: Develop specialized marketing strategies for these job categories, highlighting the benefits of term deposits for their specific financial goals.

### 3. Marital Status
- **Insight**: Single individuals are more likely to subscribe to term deposits compared to married or divorced individuals.
- **Actionable Strategy**: Create targeted campaigns for single individuals, emphasizing the advantages of saving and investing early.

### 4. Education
- **Insight**: Higher education levels, particularly tertiary education, correlate with higher subscription rates for term deposits.
- **Actionable Strategy**: Focus on educational content and financial literacy programs to attract and retain customers with higher education levels.

### 5. Balance
- **Insight**: Customers with higher account balances are more likely to subscribe to term deposits.
- **Actionable Strategy**: Offer personalized financial advice and exclusive benefits to high-balance customers to encourage term deposit subscriptions.

### 6. Housing Loan
- **Insight**: Customers without housing loans are more likely to subscribe to term deposits.
- **Actionable Strategy**: Highlight the stability and security of term deposits to customers without existing housing loans.

### 7. Contact Method
- **Insight**: Cellular contact methods are more effective in securing term deposit subscriptions compared to telephone or unknown methods.
- **Actionable Strategy**: Prioritize cellular contact methods in marketing campaigns to improve customer engagement and conversion rates.

### 8. Month
- **Insight**: Subscription rates vary by month, with higher rates observed during certain months.
- **Actionable Strategy**: Align marketing efforts with peak subscription months to maximize impact and conversion rates.

### 9. Campaign
- **Insight**: Fewer contact attempts during a campaign correlate with higher subscription rates.
- **Actionable Strategy**: Optimize the number of contact attempts to avoid customer fatigue and improve subscription rates.

### 10. Previous Campaign Outcome
- **Insight**: Customers with a successful outcome in previous campaigns are more likely to subscribe to term deposits.
- **Actionable Strategy**: Leverage data from previous successful campaigns to identify and target potential customers for new term deposit offers.

### 11. RFM Segmentation
- **Insight**: RFM (Recency, Frequency, Monetary) analysis helps identify high-value customer segments such as Champions and Potential Loyalists.
- **Actionable Strategy**: Develop tailored marketing strategies for different RFM segments to enhance customer retention and increase term deposit subscriptions.

### 12. Customer Segments
- **Insight**: Customer segments such as Champions and Potential Loyalists show higher subscription rates for term deposits.
- **Actionable Strategy**: Focus on maintaining and nurturing relationships with these high-value segments through personalized offers and exclusive benefits.




## 1. Data Preparation and Exploratory Data Analysis (EDA)
### 1.1 Data Attributes
- **Initial Data Inspection**: Loaded the dataset and checked for NaN values and duplicates.
- **Attributes Summary**: Created a summary of data types, NaN values, duplicates, and unique values for each column.

### 1.2 Feature Engineering
- **Deposit (y)**: Renamed the target variable 'y' to 'deposit' and binarized it.
- **Age**: Categorized age into life stages: young adults, middle adulthood, late adulthood, and seniors.
- **Job**: Grouped job titles into broader categories such as Management, Blue-Collar, Self-Employment, etc.
- **Balance**: Binned the balance column into quantiles.
- **Binarize Columns**: Binarized columns like default, housing, and loan.
- **Day-Month Combination**: Combined day and month columns into a single column.
- **Pdays**: Replaced -1 values in the pdays column with a placeholder.
- **One-Hot Encoding**: Applied one-hot encoding to categorical columns.

### 1.3 Client Profiling
- **Target Distribution**: Analyzed the distribution of the target variable (deposit).
- **Correlation Analysis**: Examined correlations between numerical features and the target variable.
- **Personal Information**: Analyzed the distribution of personal information features (age, job, marital status, education).
- **Banking Information**: Analyzed the distribution of banking-related features (balance, housing, loan).
- **Current Campaign**: Analyzed the distribution of features related to the current campaign (contact, month, duration, campaign).
- **Past Campaign**: Analyzed the distribution of features related to past campaigns (pdays, previous, poutcome).

## 2. RFM Segmentation
- **RFM Analysis**: Performed Recency, Frequency, and Monetary (RFM) analysis to segment customers based on their interactions with the bank.
- **Customer Segments**: Classified customers into segments such as Champions, Potential Loyalists, At-Risk Customers, etc.

## 3. Prediction
### 3.1 Feature Selection
- **RandomForestClassifier**: Trained a RandomForestClassifier to determine feature importances.
- **Feature Selection**: Selected features with importance greater than 0.01 for the final model.

### 3.2 Best Model
- **BalancedRandomForestClassifier**: Trained a BalancedRandomForestClassifier with selected features.
- **Model Evaluation**: Evaluated the model using accuracy and F2-score.

### 3.3 Hyperparameter Tuning
- **RandomizedSearchCV**: Performed hyperparameter tuning using RandomizedSearchCV to optimize the model.
- **Best Parameters**: Identified the best parameters and evaluated the model performance.


### Portfolio Impact
This project demonstrates proficiency in data analysis, feature engineering, machine learning, and model evaluation. It showcases the ability to derive actionable insights from data and develop predictive models that can drive business decisions. The comprehensive approach and detailed analysis presented in this notebook make it a valuable addition to your portfolio, highlighting your skills in data science and machine learning.


## Autor
- Samet Kurt - github: lygonian

