import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.markdown('# Check if someones a LinkedIn user')
st.write('Please enter your information')

################################# INCOME #########################################
#Get income from selection 
Income = st.selectbox("What is their Income Level:",
            options = ['Less than $10,000',
                      'under $20,000',
                      'under $30,000',
                      'under $40,000',
                      'under $50,000',
                      'under $75,000',
                      'under $100,000',
                      'under $150,000',
                      '$150,000 or more'])

#convert income to number value
if Income == 'Less than $10,000':
    Income = 1
elif Income == 'under $20,000':
    Income = 2
elif Income == 'under $30,000':
    Income = 3
elif Income == 'under $40,000':
    Income = 4
elif Income == 'under $50,000':
    Income = 5
elif Income == 'under $75,000':
    Income = 6
elif Income == 'under $100,000':
    Income = 7
elif Income == 'under $150,000':
    Income = 8
else:
    Income = 9
                    
###################################### EDUCATION ####################################

Educ = st.selectbox("What is their Education Level:",
            options = ['Less than high school (Grades 1-8 or no formal schooling)',
                      'High school incomplete (Grades 9-11 or Grade 12 with NO diploma)',
                      'High school graduate (Grade 12 with diploma or GED certificate)',
                      'Some college, no degree (includes some community college)',
                      'Two-year associate degree from a college or university',
                      'Four-year college or university degree/Bachelors degree (e.g., BS, BA, AB)',
                      'Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)',
                      'Postgraduate or professional degree, including masters, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)'])

if Educ == 'Less than high school (Grades 1-8 or no formal schooling)':
    Educ = 1
elif Educ == 'High school incomplete (Grades 9-11 or Grade 12 with NO diploma)':
    Educ = 2
elif Educ == 'High school graduate (Grade 12 with diploma or GED certificate)':
    Educ = 3
elif Educ == 'Some college, no degree (includes some community college)':
    Educ = 4
elif Educ == 'Two-year associate degree from a college or university':
    Educ = 5
elif Educ == 'Four-year college or university degree/Bachelorâs degree (e.g., BS, BA, AB)':
    Educ = 6
elif Educ == 'Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)':
    Educ = 7
else:
    Educ = 8


################################################# Parent ############################################################

Par = st.selectbox("Are they a parent of a child under 18 living in their home?:",
                        options = ['Yes',
                                   'No'])

if Par == 'Yes':
    Par = 1
else:
    Par = 2

################################################# MARITAL ####################################################

Mar = st.selectbox("What is their marital status:",
            options = ['Married',
                      'Living with a partner',
                      'Divorced',
                      'Separated',
                      'Widowed',
                      'Never been married'])

if Mar == 'Married':
    Mar = 1
elif Mar == 'Living with a partner':
    Mar = 2
elif Mar == 'Divorced':
    Mar = 3
elif Mar == 'Seperated':
    Mar = 4
elif Mar == 'Widowed':
    Mar = 5
else:
    Mar = 6

 ################################################# Gender ####################################################

Gender = st.selectbox("What is their Gender?:",
                        options = ['Male',
                                   'Female',
                                   'Other'])

if Gender == 'Male':
    Gender = 1
elif Gender == 'Female':
    Gender = 2
else:
    Gender = 3

 ################################################# Age ####################################################

Age = st.slider(label = 'Enter their age:',
                min_value = 1,
                max_value = 98,
                value = 23)



#Read in data 
s = pd.read_csv('social_media_usage.csv')

#Create function to make target column binary
def cleansm(x):
    true_index = np.where(x == 1)
    if true_index[0] == 0:
        x = 1 
    else:
        x = 0 
    return x


#Make anything that's not 1 0 in target column
for i in s.index:
    s.at[i, 'web1h'] = cleansm(s.at[i, 'web1h'])

s['sm_li'] = s.web1h

#Create new df with relevant columns 
ss = s[['sm_li', 'income', 'educ2', 'par', 'marital', 'gender', 'age']].copy()



#Get rid of nulls
ss = ss[ss.income <= 9]
ss = ss[ss.educ2 <= 8]
ss = ss[ss.age <= 98]


#Train test split
y = ss.sm_li
X = ss.drop(columns=['sm_li'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Fit the model
model = LogisticRegression(class_weight = 'balanced')
model.fit(X_train, y_train)


pred1 = {'income' : [Income], 'educ2': [Educ],'par' : [Par],'marital':  [Mar],'gender':  [Gender],'age': [Age]}
pred1_df = pd.DataFrame(data = pred1)

pred_output = model.predict(pred1_df)

pred_prob = model.predict_proba(pred1_df)

if pred_output == 1:
    result = "This Person is a LinkedIn user"
else:
    result = "This Person is not a LinkedIn user"

st.write(result)

prob_result = 'The probability that they use LinkedIn is ' + str(round(pred_prob[0][1] * 100)) + '%'

st.write(prob_result)

