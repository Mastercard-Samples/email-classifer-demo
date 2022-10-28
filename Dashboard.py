#!/usr/bin/env python
# coding: utf-8

import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
from datetime import datetime
from ops_modules.parse_email import ParseMailData
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

test_data = 'ML_testData/test_data/test_data.CSV'

## Load the best ML model
# Knn
with open('ML_algorithms/Models/best_knn.pickle', 'rb') as data:
    knn = pickle.load(data)

# features_test
with open('Pickles/features_test.pickle', 'rb') as data:
    features_test = pickle.load(data)

## Load TF-IDF model
with open('Pickles/tfidf.pickle', 'rb') as data:
    tfidf = pickle.load(data)

## Loading the test data prior to the current month
with open('ML_testData/Predictions/ML_data.pickle',  'rb') as data:
    ML_test_data = pickle.load(data)

## Read the test data
df = pd.read_csv(test_data, encoding="ISO-8859-1")
df = df.fillna('dummy')

## Parsing the mail data to add day, year, month and date columns
parse_obj = ParseMailData(test_data)
parse_df = parse_obj.parse()
current_year = datetime.now().year
current_month = datetime.now().strftime('%h')

## Fetching the current month data 
parse_df_current = parse_df.loc[(parse_df['Month'] == (pd.Period(datetime.now(), 'M') - 1).strftime('%b')) & (parse_df['Year'] == current_year)]
## Specify Month(str) as "Nov" and Year(int) as 2021
# parse_df_current = parse_df.loc[(parse_df['Month'] == "May") & (parse_df['Year'] == 2022)]
parse_df_current = parse_df_current.reset_index(drop=True)


def feature_creation(content_parsed):
    # Removing \r \n and extra spaces
    content_parsed['Content_Parsed_1'] = content_parsed['Body'].str.replace("\r", " ")
    content_parsed['Content_Parsed_1'] = content_parsed['Content_Parsed_1'].str.replace("\n", " ")
    content_parsed['Content_Parsed_1'] = content_parsed['Content_Parsed_1'].str.replace("    ", " ")
    # content_parsed['Category'] = category_subject_df['Category']

    # Removing " when quoting text
    content_parsed['Content_Parsed_1'] = content_parsed['Content_Parsed_1'].str.replace('"', '')

    # Lowercasing the text
    content_parsed['Content_Parsed_1'] = content_parsed['Content_Parsed_1'].str.lower()

    # Removing common non-relevant occuring words
    ignore_words = ['mastercard', 'com', 'senior', 'software', 'engineer', 'mountainview', 'central', 'park', 'leopardstown',
                    'dublin', '18', 'ireland', 'cc', 'subject', 'mailto', 'api_consultancy_and_standards', 'api_onboarding']
    for ig_word in ignore_words:
        content_parsed['Content_Parsed_1'] = content_parsed['Content_Parsed_1'].str.replace(ig_word, ' ')

    # Removing punctuation signs and other unwanted symbols
    punctuation_signs = list("?:!.,;<>|@")

    for punct_sign in punctuation_signs:
        content_parsed['Content_Parsed_1'] = content_parsed['Content_Parsed_1'].str.replace(punct_sign, ' ')

    # Removing possessive nouns
    content_parsed['Content_Parsed_1'] = content_parsed['Content_Parsed_1'].str.replace("'s", " ")
    
    ##### Lemmatization #####
    # Saving the lemmatizer into an object
    wordnet_lemmatizer = WordNetLemmatizer()
    lemma_text_list = []

    for row in range(0, len(content_parsed)):

        # Create an empty list containing lemmatized words
        lemma_list = []

        # Save the text and its words into an object
        text = content_parsed.loc[row]['Content_Parsed_1']
        text_words = text.split(" ")

        # Iterate through every word to lemmatize
        for word in text_words:
            lemma_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))

        # Join the list
        lemma_text = " ".join(lemma_list)

        # Append to the list containing the texts
        lemma_text_list.append(lemma_text)

    content_parsed['parsed_lemmatized_text'] = lemma_text_list
    
    ##### Stop words removal #####
    stop_words = list(stopwords.words('english'))
    # Adding the stopwords from the TO: From: CC: column
    # which is all the names

    stop_words_df = pd.DataFrame({})
    stop_words_df['From: (Name)'] = df['From: (Name)']
    stop_words_df['To: (Name)'] = df['To: (Name)']
    stop_words_df['CC: (Name)'] = df['CC: (Name)']
    #stop_words_df = df_distinct.iloc[:, :2]

    for column in stop_words_df:
        # Lowercasing the text
        stop_words_df[column] = stop_words_df[column].str.lower()

        # Removing punctuation signs and other unwanted symbols
        stop_words_df[column] = stop_words_df[column].str.replace('dummy', '')

        # Removing punctuation signs and other unwanted symbols
        stop_words_punctuation_signs = list(",;)(")

        for stop_words_punct_sign in stop_words_punctuation_signs:
            stop_words_df[column] = stop_words_df[column].str.replace(stop_words_punct_sign, ' ')
            
    word = list()
    for i in range(0, len(stop_words_df)):
        for j in stop_words_df.loc[i].values:
            word.append(j.split())
    new_stop_words = [item for sublist in word for item in sublist]


    stop_words_unique = set(new_stop_words) # To get unique values
    
    stop_words.extend(stop_words_unique)
    remove_words = ['jamstack', 'onboarding', 'support', 'product', 'operations', 'api', 'apis', 'project', 
                    'architecture', 'security', 'development', 'key', 'jenkins', 'dev', 'external', 'team', 'digital',
                    'helpdesk', 'axon', 'gateway', 'xmlgw', 'access', 'ping', 'strategic', 'developers', 'postgres',
                    'management', 'xml', 'gw', 'service', 'dba', 'standards']

    for w in remove_words:
        if w in stop_words:
            stop_words.remove(w)
        
    # Takes time to process 5-6 mins.
    # This is to remove all the stopwords from the Body.
    content_parsed['stop_words_parsed'] = content_parsed['parsed_lemmatized_text']
    for stop_word in stop_words:
        if (stop_word == '?ukasz'):
            stop_word = '\?ukasz'
        regex_stopword = r"\b" + stop_word + r"\b"
        content_parsed['stop_words_parsed'] = content_parsed['stop_words_parsed'].str.replace(regex_stopword, '')

    # Removing the unwanted columns
    content_parsed = content_parsed.drop(['Content_Parsed_1', 'parsed_lemmatized_text'], axis=1)
    
    # Renaming the parsed column
    content_parsed = content_parsed.rename(columns={'stop_words_parsed': 'Content_Parsed'})
    
    # TF-IDF
    features = tfidf.transform(content_parsed['Content_Parsed']).toarray()
    
    return features, content_parsed


def get_category_name(category_id):
    category_codes = {'1' : 'Service Proxy troubleshooting / APIGW', 
                      '2' : 'Onboarding generic queries',
                      '3' : 'Assessment/rescore queries/early spec/exception requests',
                      '4' : 'Access to Tool queries', 
                      '5' : 'API Standards queries',
                      '6' : 'zally',
                      '7' : 'Client libs', 
                      '8' : 'Jamstack content reviewer',
                      '9' : 'Axon Queries',
                      '10': 'Mastercard Developers Notification'
                      }
    for cid, cname  in category_codes.items():    
        if cid == category_id:
            return cname


def predict_from_features(features):
        
    # Obtain the highest probability of the predictions for each mail
    predictions_proba = knn.predict_proba(features).max(axis=1)    
    
    # Predict using the input model
    predictions_pre = knn.predict(features)

    # Replace prediction with 6 if associated cond. probability less than threshold
    predictions = []

    for prob, cat in zip(predictions_proba, predictions_pre):
        if prob > .65:
            predictions.append(cat)
        else:
            predictions.append(5)

    # Return result
    categories = [get_category_name(x) for x in predictions]
    
    return categories


def complete_df(df, categories):
    df['Prediction'] = categories
    return df


# 6-7 mins
# Features creation
features, df_show_info = feature_creation(parse_df_current)

## Cancatenating the current month's test data with the previous data
ML_test_data = pd.concat([ML_test_data, parse_df_current], ignore_index=True)
ML_test_data = ML_test_data.drop_duplicates(subset=['Subject']).reset_index(drop=True)

## Predict
predictions = predict_from_features(features)

## Put into dataset
df_predictions_current = complete_df(df_show_info, predictions)


# Appending the current predictions with the previous data
with open('ML_testData/Predictions/knn_test.pickle', 'rb') as data:
    previous_data = pickle.load(data)

# Appending the current predictions with the previous data
total_data = pd.concat([previous_data, df_predictions_current], ignore_index=True)
total_data = total_data.drop_duplicates(subset=['Subject'])
total_data = total_data.reset_index(drop=True)


# Saving ML_test_data with updated values in df pickle file
with open('ML_testData/Predictions/ML_data.pickle', 'wb') as output:
    pickle.dump(ML_test_data, output)

# Saving predicted values in df pickle file    
with open('ML_testData/Predictions/knn_test.pickle', 'wb') as output:
    pickle.dump(total_data, output)

    
##### Implementing the dashboard

# Knn
with open('ML_testData/Predictions/knn_test.pickle', 'rb') as data:
    parse_df = pickle.load(data)

## Creating DF for monthwise data and plotting graph

df_mod = pd.DataFrame({})
df_mod['Date'] = parse_df['Date']
df_mod['Month'] = parse_df['Month']
df_mod['Year'] = parse_df['Year']
df_mod['Prediction'] = parse_df['Prediction']

# Sort values by date
df_mod = df_mod.sort_values(by='Date').reset_index(drop=True)

# Grouping and summing the values for each category
df_grouped = pd.crosstab([df_mod['Month'], df_mod['Year']], df_mod.Prediction).reset_index().rename_axis(None, axis=1)
df_grouped_year = df_grouped.groupby(by='Year')
df_grouped_2020 = df_grouped_year.get_group(2020)
df_grouped_2021 = df_grouped_year.get_group(2021)
df_grouped_2022 = df_grouped_year.get_group(2022)

########## Graph 1: plotting the sum of the queries for each month yearwise
##### YEAR - 2020
# not using in graph at the moment
# Sorting based on Month. First converting to categorical month equivalent.
df_grouped_2020["Month"] = pd.to_datetime(df_grouped_2020.Month, format='%b', errors='coerce').dt.month
df_grouped_2020 = df_grouped_2020.sort_values(by="Month")
# YEAR - 2020
# not using in graph at the moment 
# Converting back to Month name from categorical equivalent value.
# 9 categories at the moment
df_grouped_2020['Month'] = pd.to_datetime(df_grouped_2020.Month, format='%m', errors='coerce').dt.month_name().str.slice(stop=3)
df_grouped_2020['Sum'] = df_grouped_2020.iloc[:,-10:].sum(axis=1)
df_grouped_2020 = df_grouped_2020.reset_index(drop=True)

## Plotting graph 2020
fig_count_year_2020 = px.bar(df_grouped_2020, x="Month", y="Sum", color="Month", height=500, width=1400)

##### YEAR - 2021
# not using in graph at the moment
# Sorting based on Month. First converting to categorical month equivalent.
df_grouped_2021["Month"] = pd.to_datetime(df_grouped_2021.Month, format='%b', errors='coerce').dt.month
df_grouped_2021 = df_grouped_2021.sort_values(by="Month")
# YEAR - 2021
# not using in graph at the moment 
# Converting back to Month name from categorical equivalent value.
df_grouped_2021['Month'] = pd.to_datetime(df_grouped_2021.Month, format='%m', errors='coerce').dt.month_name().str.slice(stop=3)
df_grouped_2021['Sum'] = df_grouped_2021.iloc[:,-10:].sum(axis=1)
df_grouped_2021 = df_grouped_2021.reset_index(drop=True)

## Plotting graph 2021
fig_count_year_2021 = px.bar(df_grouped_2021, x="Month", y="Sum", color="Month", height=500, width=1400)

##### YEAR - 2022
# not using in graph at the moment
# Sorting based on Month. First converting to categorical month equivalent.
df_grouped_2022["Month"] = pd.to_datetime(df_grouped_2022.Month, format='%b', errors='coerce').dt.month
df_grouped_2022 = df_grouped_2022.sort_values(by="Month")
# YEAR - 2022
# not using in graph at the moment 
# Converting back to Month name from categorical equivalent value.
df_grouped_2022['Month'] = pd.to_datetime(df_grouped_2022.Month, format='%m', errors='coerce').dt.month_name().str.slice(stop=3)
df_grouped_2022['Sum'] = df_grouped_2022.iloc[:,-10:].sum(axis=1)
df_grouped_2022 = df_grouped_2022.reset_index(drop=True)

## Plotting graph 2022
fig_count_year_2022 = px.bar(df_grouped_2022, x="Month", y="Sum", color="Month", height=500, width=1400)

########## Graph 2: plotting the sum of the queries for each month yearwise and comparing with the last year.
df_grouped_sum = df_grouped.sort_values(by="Year")
df_grouped_sum['Sum'] = df_grouped.iloc[:,-10:].sum(axis=1)

# Sorting based on Month. First converting to categorical month equivalent.
df_grouped_sum["Month"] = pd.to_datetime(df_grouped_sum.Month, format='%b', errors='coerce').dt.month
df_grouped_sum = df_grouped_sum.sort_values(by="Month")
df_grouped_sum['Month'] = pd.to_datetime(df_grouped_sum.Month, format='%m', errors='coerce').dt.month_name().str.slice(stop=3)
df_grouped_sum = df_grouped_sum.reset_index(drop=True)

## Plotting graph
fig_grouped_sum = go.Figure()
fig_grouped_sum.add_trace(go.Bar(x=df_grouped_2020['Month'],
                y=df_grouped_2020['Sum'],
                name='2020',
                marker_color='rgb(55, 83, 109)'
                ))
fig_grouped_sum.add_trace(go.Bar(x=df_grouped_2021['Month'],
                y=df_grouped_2021['Sum'],
                name='2021',
                marker_color='rgb(26, 118, 255)'
                ))
fig_grouped_sum.add_trace(go.Bar(x=df_grouped_2022['Month'],
                y=df_grouped_2022['Sum'],
                name='2022',
                marker_color='rgb(150, 75, 0)'
                ))

fig_grouped_sum.update_layout(
    title='Year wise total count',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='Count',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1, # gap between bars of the same location coordinate.
    height=500,
    width=1400
)

########## Graph 3: plotting the sum of the individual queries for each month yearwise.

df_mod_group = df_mod.groupby(by='Year')
df_2020 = df_mod_group.get_group(2020)
df_2021 = df_mod_group.get_group(2021)
df_2022 = df_mod_group.get_group(2022)
df_2020 = df_2020.reset_index(drop=True)
df_2021 = df_2021.reset_index(drop=True)
df_2022 = df_2022.reset_index(drop=True)

# YEAR - 2020
fig_2020 = px.bar(df_2020,
                  x="Month",
                  color="Prediction",
                  facet_col="Prediction",
                  facet_col_wrap=2,
                  width=1600,
                  height=1600)

# YEAR - 2021
fig_2021 = px.bar(df_2021, x="Month", 
                  color="Prediction",
                  facet_col="Prediction",
                  facet_col_wrap=2,
                  width=1600, 
                  height=1600)

# YEAR - 2022
fig_2022 = px.bar(df_2022, x="Month", 
                  color="Prediction",
                  facet_col="Prediction",
                  facet_col_wrap=2,
                  width=1600, 
                  height=1600)

########## Graph 4: plotting the Timeseries Area graph for each category.

## Creating Timeseries Area graph for categories
df_mod_datewise = pd.get_dummies(df_mod.set_index('Date').Prediction).sum(level=0).reset_index()

## Pltting graph
fig_sub_category_daily = make_subplots(rows=5, cols=2)

fig_sub_category_daily.add_trace(go.Scatter(name='API Standards queries', x=df_mod_datewise.Date, y=df_mod_datewise['API Standards queries'], fill='tozeroy'), row=1, col=1)
fig_sub_category_daily.add_trace(go.Scatter(name='Access to Tool queries', x=df_mod_datewise.Date, y=df_mod_datewise['Access to Tool queries'], fill="tonexty"), row=2, col=1)
fig_sub_category_daily.add_trace(go.Scatter(name='Assessment/rescore/early spec/exception requests', x=df_mod_datewise.Date, y=df_mod_datewise['Assessment/rescore queries/early spec/exception requests'], fill='tozeroy'), row=3, col=1)
fig_sub_category_daily.add_trace(go.Scatter(name='Client libs', x=df_mod_datewise.Date, y=df_mod_datewise["Client libs"], fill="tonexty"), row=4, col=1)
fig_sub_category_daily.add_trace(go.Scatter(name='Jamstack content reviewer', x=df_mod_datewise.Date, y=df_mod_datewise['Jamstack content reviewer'], fill='tozeroy'), row=1, col=2)
fig_sub_category_daily.add_trace(go.Scatter(name='Onboarding generic queries', x=df_mod_datewise.Date, y=df_mod_datewise["Onboarding generic queries"], fill="tonexty"), row=2, col=2)
fig_sub_category_daily.add_trace(go.Scatter(name='Service Proxy troubleshooting / APIGW', x=df_mod_datewise.Date, y=df_mod_datewise['Service Proxy troubleshooting / APIGW'], fill='tozeroy'), row=3, col=2)
fig_sub_category_daily.add_trace(go.Scatter(name='zally', x=df_mod_datewise.Date, y=df_mod_datewise["zally"], fill="tonexty"), row=4, col=2)
fig_sub_category_daily.add_trace(go.Scatter(name='Axon Queries', x=df_mod_datewise.Date, y=df_mod_datewise["Axon Queries"], fill="tonexty"), row=5, col=1)
fig_sub_category_daily.add_trace(go.Scatter(name='Mastercard Developers Notification', x=df_mod_datewise.Date, y=df_mod_datewise["Mastercard Developers Notification"], fill="tonexty"), row=5, col=2)

fig_sub_category_daily.update_layout(width=1600, height=1300, title_text="Predictions category wise")


########## Graph 5: plotting the Timeseries Area graph with total count for each category

fig_subplots = make_subplots(rows=5, cols=2)
df_mod_pro = df_mod[df_mod.Prediction == 'Service Proxy troubleshooting / APIGW']
df_mod_api = df_mod[df_mod.Prediction == 'API Standards queries']
df_mod_onb = df_mod[df_mod.Prediction == 'Onboarding generic queries']
df_mod_zal = df_mod[df_mod.Prediction == 'zally']
df_mod_jam = df_mod[df_mod.Prediction == 'Jamstack content reviewer']
df_mod_res = df_mod[df_mod.Prediction == 'Assessment/rescore queries/early spec/exception requests']
df_mod_cli = df_mod[df_mod.Prediction == 'Client libs']
df_mod_acc = df_mod[df_mod.Prediction == 'Access to Tool queries']
df_mod_aq = df_mod[df_mod.Prediction == 'Axon Queries']
df_mod_mdn = df_mod[df_mod.Prediction == 'Mastercard Developers Notification']
fig_subplots.add_trace(go.Scatter(name='Service Proxy troubleshooting / APIGW', x=df_mod_pro.Date, fill='tozeroy'), row=1, col=1)
fig_subplots.add_trace(go.Scatter(name='API Standards queries', x=df_mod_api.Date, fill='tozeroy'), row=2, col=1)
fig_subplots.add_trace(go.Scatter(name='Onboarding generic queries', x=df_mod_onb.Date, fill='tozeroy'), row=3, col=1)
fig_subplots.add_trace(go.Scatter(name='zally', x=df_mod_zal.Date, fill='tozeroy'), row=4, col=1)
fig_subplots.add_trace(go.Scatter(name='Jamstack content reviewer', x=df_mod_jam.Date, fill='tozeroy'), row=1, col=2)
fig_subplots.add_trace(go.Scatter(name='Assessment/rescore/early spec/exception requests', x=df_mod_res.Date, fill='tozeroy'), row=2, col=2)
fig_subplots.add_trace(go.Scatter(name='Client libs', x=df_mod_cli.Date, fill='tozeroy'), row=3, col=2)
fig_subplots.add_trace(go.Scatter(name='Access to Tool queries', x=df_mod_acc.Date, fill='tozeroy'), row=4, col=2)
fig_subplots.add_trace(go.Scatter(name='Axon Queries', x=df_mod_aq.Date, fill='tozeroy'), row=5, col=1)
fig_subplots.add_trace(go.Scatter(name='Mastercard Developers Notification', x=df_mod_mdn.Date, fill='tozeroy'), row=5, col=2)

fig_subplots.update_layout(autosize=False, width=1600, height=1300)


########## Table 1: Creating the Test Data for Visualisation Table
df_table = pd.DataFrame({})
df_table['Categories']  = parse_df['Prediction'].value_counts().index[:10].tolist()
df_table['Count'] = parse_df['Prediction'].value_counts().tolist()
df_table['Percentage'] = parse_df['Prediction'].value_counts(normalize=True).tolist()

#################### Storing the DataFrame's as pickles ####################

# 
with open("ML_testData/Dash_visualisations/df_mod.pickle", 'wb') as output:
    pickle.dump(df_mod, output)

#
with open("ML_testData/Dash_visualisations/df_grouped.pickle", 'wb') as output:
    pickle.dump(df_grouped, output)
    
#
with open("ML_testData/Dash_visualisations/df_grouped_2020.pickle", 'wb') as output:
    pickle.dump(df_grouped_2020, output)
    
#
with open("ML_testData/Dash_visualisations/df_grouped_2021.pickle", 'wb') as output:
    pickle.dump(df_grouped_2021, output)
    
#
with open("ML_testData/Dash_visualisations/df_grouped_2022.pickle", 'wb') as output:
    pickle.dump(df_grouped_2022, output)
    
#
with open("ML_testData/Dash_visualisations/df_grouped_sum.pickle", 'wb') as output:
    pickle.dump(df_grouped_sum, output)

############################ Dash Implementation ############################

# Stylesheet
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Colors
colors = {
    'background': '#ffffff',
    'text': '#696969',
    'header_table': '#ffedb3'
}

# Markdown text
markdown_text1 = '''

This application uses the dataset for Email from API C&S mailbox, 
predicts their category and then shows a graphic summary.

KNN machine model is used to predict the categories.

'''

# markdown_text2 = '''
# 1. API Standards queries 
# 2. Assessment/rescore queries/early spec/exception requests
# 3. Access to Tool queries
# 4. Onboarding generic queries
# 5. Service Proxy troubleshooting / APIGW
# 6. zally
# 7. Jamstack content reviewer
# 8. Client libs
# '''
markdown_text2 = '''

'''

app.layout = html.Div(style={'backgroundColor':colors['background']}, children=[
    
    # Title
    html.H2(children='API C&S Email Classification App',
            style={
                'textAlign': 'left',
                'color': colors['text'],
                'padding': '10px',
                'backgroundColor': colors['header_table']

            },
            className='banner',
            ),

    # Sub-title Left
    html.Div([
        dcc.Markdown(children=markdown_text1)],
        style={'width': '49%', 'display': 'inline-block'}),
    
    # Sub-title Right
    html.Div([
        dcc.Markdown(children=markdown_text2)],
        style={'width': '49%', 'float': 'right', 'display': 'inline-block'}),

    # Space between text and dropdown
    html.H1(id='space', children=' '),

    # Dropdown
    html.Div([
        dcc.Dropdown(
            id='dropdown',
            options=[
                {'label': 'Overall', 'value': 'Overall'},
                {'label': 'Category Daily Count', 'value': 'Category Daily Count'},
                {'label': 'Total Count', 'value': 'Total Count'},
                {'label': 'Total Count Category Wise', 'value': 'Total Count Category Wise'},
                {'label': 'Year 2020', 'value': 'Year 2020'},
                {'label': 'Year 2020 Cumulative', 'value': 'Year 2020 Cumulative'},
                {'label': 'Year 2021', 'value': 'Year 2021'},
                {'label': 'Year 2021 Cumulative', 'value': 'Year 2021 Cumulative'},
                {'label': 'Year 2022', 'value': 'Year 2022'},
                {'label': 'Year 2022 Cumulative', 'value': 'Year 2022 Cumulative'}
            ],
            value='Overall',
            clearable=False,
        )],
    ),

    # Graph1
    html.Div([
        dcc.Slider(id='width', min=1400, max=2200, step=100, value=1600,
                marks={x: str(x) for x in [1400, 1600, 1800, 2000, 2200]}),
        dcc.Graph(id='graph1')]),
    
    # Graph3
    html.Div([
        dcc.Graph(id='graph3')]),
    
    # Table title
    html.Div(id='table-title', children='Summary of emails:'),

    # Space
    html.H1(id='space2', children=' '),
    
    # Table
    html.Div([
        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in ['Categories', 'Count', 'Percentage']],
            style_data={'whiteSpace': 'normal'},
            style_as_list_view=True,
            style_cell={'padding': '5px', 'textAlign': 'left', 'backgroundColor': colors['background']},
            style_header={
                'backgroundColor': colors ['header_table'],
                'fontWeight': 'bold'
            },
            style_table={
                'maxHeight': '300',
                'overflowY':'scroll'
            },
            css=[{
                'selector': '.dash-cell div.dash-cell-value',
                'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
            }]
        )],
        style={'width': '75%','float': 'left', 'position': 'relative', 'left': '12.5%', 'display': 'inline-block'}),    
        
#     # Hidden div inside the app that stores the intermediate value
#     html.Div(id='intermediate-value', style={'display': 'none'})
])

@app.callback(
    Output('graph1', 'figure'),
    [Input('dropdown', 'value'),
      Input('width', 'value')])
def update_barchart(data, width):
    if (data == "Overall"):
        predictions = parse_df['Prediction'].value_counts()
        fig = px.bar(predictions, 
                      x="Prediction",
                      height=500,
                      color='Prediction'
                    )
        return fig
    elif (data == "Category Daily Count"):
        fig_sub_category_daily.update_layout(width=int(width))
        return fig_sub_category_daily
    elif (data == "Total Count Category Wise"):
        fig_subplots.update_layout(width=int(width))
        return fig_subplots
    elif (data == "Total Count"):
        fig_grouped_sum.update_layout(width=int(width))
        return fig_grouped_sum
    elif (data == "Year 2020"):
        fig_2020.update_layout(width=int(width))
        return fig_2020
    elif (data == "Year 2020 Cumulative"):
        return fig_count_year_2020
    elif (data == "Year 2021"):
        fig_2021.update_layout(width=int(width))
        return fig_2021
    elif (data == "Year 2021 Cumulative"):
        return fig_count_year_2021
    elif (data == "Year 2022"):
        fig_2021.update_layout(width=int(width))
        return fig_2022
    elif (data == "Year 2022 Cumulative"):
        return fig_count_year_2022
    
@app.callback(
    Output('graph3', 'figure'),
    [Input('dropdown', 'value')])
def update_piechart(data):
    predictions = parse_df['Prediction'].value_counts()
    fig = px.pie(predictions,
                  values='Prediction',
                  names=parse_df['Prediction'].value_counts().index[:10].tolist() # Sorting the names based on count
                )
    return fig
    
@app.callback(
    Output('table', 'data'),
    [Input('dropdown', 'value')])
def update_table(data):
    data = df_table.to_dict('records')
    return data
    
# Loading CSS
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})

app.run_server(host='0.0.0.0', port=8080, debug=True)
