# %%
#Testing Python Works
print('hello world')

# %%
#Building a Linear Regression Model to Predict Yearly Amount Spent by an Ecommerce Platform Visitors
#The model predicts how much a customer spends on an ecommerce platform per year based on parameters like Time on Phone App,Time on Website,Average Session Length and Length of Membership

# %%
#Import all necessary libraries
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

# %%
#We now load our dataset
df = pd.read_csv("archive/ecommerce.csv")



# %%
#We now build a Linear Regression Model Using the variable(Assume one variable at the moment) Length of Membership again Yearly Amount Spent
sns.lmplot(
    data=df,
    x="Length of Membership",
    y="Yearly Amount Spent",
    scatter_kws={'alpha': 0.5}
)

# %%
#Now we move to creating our Linear Regression Model with all the variables considered

# %%
#We start by importing the train,test and split commands in scikit-learn
from sklearn.model_selection import train_test_split

# %%
#We set the variables and output of the model
#Note that the capitalization in x variable is to represent an array of variables
X= df[['Avg. Session Length', 'Time on App', 'Time on Website','Length of Membership']]
y= df['Yearly Amount Spent']

# %%
#Now we split the dataset into the respective training and testing segments
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.32, random_state=42)


# %%
#Training the model

# %%
from sklearn.linear_model import LinearRegression
lm= LinearRegression()

# %%
#Pass in the training dataset portion
lm.fit(X, y)








