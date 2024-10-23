
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = 'vscode+png'
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import os


df_banking_raw = pd.read_csv('data/banking_train.csv', sep=';')
df_banking = df_banking_raw.copy()
df_banking = df_banking.rename(columns={'y': 'deposit'})
df_banking['deposit'] = df_banking['deposit'].replace({'no': int(0), 'yes': int(1)})
df_banking['deposit'] = df_banking['deposit'].astype(int)

def binarize_columns(df, columns):
    """
    This function takes a DataFrame and a list of column names, binarizes the specified columns,
    and returns the DataFrame with the binarized columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): The list of columns to be binarized.
    df_binarized = binarize_columns(df_banking, ['default', 'housing', 'loan', 'y'])

    Returns:
    pd.DataFrame: The DataFrame with binarized columns.
    """
    df = df.copy()
    for column in columns:
        df[column] = df[column].apply(lambda x: int(1) if x == 'yes' else int(0))
    return df

def substitute_pdays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Substitute the value of 'pdays' column in the dataframe.
    
    This function replaces all occurrences of -1 in the 'pdays' column with 10000.
    
    Parameters:
    df (pd.DataFrame): The input dataframe containing a 'pdays' column.
    example: df_temp = substitute_pdays(df_temp)
    Returns:
    pd.DataFrame: The dataframe with substituted 'pdays' values.
    """
    df['pdays'] = df['pdays'].replace(-1, 10000)
    return df


def calculate_rfm(df):
    """
    Calculate df (Recency, Frequency, Monetary) metrics for the given DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing customer data.

    Returns:
    pd.DataFrame: The DataFrame with df metrics.
    """
    # Recency: The number of days since the last contact
    df['Recency'] = df['pdays'].apply(lambda x: 0 if x == 10000 else x)
    # Frequency: The number of contacts performed during this campaign and for this client
    df['Frequency'] = df['campaign'] + df['previous']
    # Monetary: The balance of the customer
    df['Monetary'] = df['balance']


    # Assigning R_Score
    df['R_Score'] = 1  # Assign a score of 1 to all Recency values of 0
    non_zero_mask = df['Recency'] != 0    # Create a mask for non-zero Recency values
    df.loc[non_zero_mask, 'R_Score']
    df.loc[non_zero_mask, 'R_Score'] = pd.qcut(df.loc[non_zero_mask, 'Recency'], q=3, labels=[4, 3, 2])  # Apply pd.qcut to non-zero Recency values

    # Assigning F_Score
    df["F_Score"] = pd.qcut(df["Frequency"], q=5, labels=[1, 2, 3, 4], duplicates='drop')

    # Assigning M_Score
    df["M_Score"] = pd.qcut(df["Monetary"], q=4, labels=[1, 2, 3, 4])

    # Combine RFM scores
    df['RFM_Score'] = df['R_Score'].astype(int) + df['F_Score'].astype(int) + df['M_Score'].astype(int)
    
    # Create df segments based on the RFM score
    segment_labels = ['Low-Value', 'Mid-Value', 'High-Value']
    df['Value_Segment'] = pd.qcut(df['RFM_Score'], q=3, labels=segment_labels)
    # Convert the Value Segment column to a categorical type with the specified order
    df['Value_Segment'] = pd.Categorical(df['Value_Segment'], categories=segment_labels, ordered=True)
    
    # Assign df segments based on the df score
    df['Customer_Segments'] = ''
    df.loc[df['RFM_Score'] >= 9, 'Customer_Segments'] = 'Champions'
    df.loc[(df['RFM_Score'] >= 6) & (df['RFM_Score'] < 9), 'Customer_Segments'] = 'Potential Loyalists'
    df.loc[(df['RFM_Score'] >= 5) & (df['RFM_Score'] < 6), 'Customer_Segments'] = 'At Risk Customers'
    df.loc[(df['RFM_Score'] >= 4) & (df['RFM_Score'] < 5), 'Customer_Segments'] = "Can't Lose"
    df.loc[(df['RFM_Score'] >= 3) & (df['RFM_Score'] < 4), 'Customer_Segments'] = "Lost"
    
    df['Customer_Segments'] = pd.Categorical(df['Customer_Segments'], categories=['Champions', 'Potential Loyalists', "Can't Lose", 'At Risk Customers', 'Lost'], ordered=True)
    return df


# %%
df_temp = df_banking.copy()
df_temp = substitute_pdays(df_temp)
df_temp = calculate_rfm(df_temp)
df_temp = binarize_columns(df_temp, ["loan", "housing", "default"])
df_temp.poutcome = df_temp.poutcome.replace({'unknown': 'Unknown', 'failure': 'Failure', 'success': 'Success', "other": "Unknown"})
df_temp.month = pd.Categorical(df_temp.month, categories=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], ordered=True) 

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server
# Define the layout of the app
app.layout = html.Div([
    dcc.RadioItems(
        id='deposit-filter',
        options=[
            {'label': 'All', 'value': 'All'},
            {'label': 'Subscribers', 'value': '1'},
            {'label': 'non-Subscribers', 'value': '0'}
        ],
        value='All',
        labelStyle={'display': 'inline-block'}
    ),
    dcc.Tabs(id="tabs", value='All', children=[
        dcc.Tab(label='All', value='All'),
        dcc.Tab(label='Champions', value='Champions'),
        dcc.Tab(label='Potential Loyalists', value='Potential Loyalists'),
        dcc.Tab(label="Can't Lose", value="Can't Lose"),
        dcc.Tab(label='At Risk Customers', value='At Risk Customers'),
        dcc.Tab(label='Lost', value='Lost'),
    ]),
    html.Div(id='tabs-content')
])

# Define the callback to update the content based on the selected tab and deposit filter
@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value'),
     Input('deposit-filter', 'value')]
)
def render_content(tab, deposit_filter):
    if tab == 'All':
        filtered_df = df_temp
    else:
        filtered_df = df_temp[df_temp['Customer_Segments'] == tab]

    if deposit_filter == '1':
        filtered_df = filtered_df[filtered_df['deposit'] == 1]
    elif deposit_filter == '0':
        filtered_df = filtered_df[filtered_df['deposit'] == 0]

    # Create the figure with filtered data
    fig = make_subplots(
        rows=5, cols=8,
        specs=[
            [{'rowspan': 1, 'colspan': 2}, None, {"type": "domain", 'rowspan': 2, 'colspan': 2}, None, {'rowspan': 1, 'colspan': 2}, None, {'rowspan': 1, 'colspan': 2}, None],
            [{"type": "domain", 'rowspan': 4, 'colspan': 2}, None, None, None, {'rowspan': 2, 'colspan': 2}, None, None, {'rowspan': 1, 'colspan': 1}],
            [None, None, {'rowspan': 1, 'colspan': 2}, None, None, None, None, {'rowspan': 1, 'colspan': 1}],
            [None, None, {"type": "domain", 'rowspan': 2, 'colspan': 2}, None, {"type": "domain", 'rowspan': 2, 'colspan': 2}, None, None, {'rowspan': 1, 'colspan': 1}],
            [None, None, None, None, None, None, None, {'rowspan': 1, 'colspan': 1}],
        ],
        subplot_titles=["Age [year]", "Contact", "Duration [sec]", "Banking [€]", "Job", "Number of contacts", "", "Month of Call", "", "Education", "Marital", "", ""]
    )

    # Add traces to the subplots based on the specs

    # Histogram
    fig.add_trace(go.Histogram(x=filtered_df.month.sort_values(), name="", showlegend=False, opacity=0.4, marker_color=px.colors.qualitative.Prism[9]), 3, 3)  # month

    # Pie
    fig.add_trace(go.Pie(labels=filtered_df['contact'].value_counts().index, values=filtered_df['contact'].value_counts().values, showlegend=False,
                         hole=.4, textinfo='label+percent', textposition='inside', insidetextorientation='radial',
                         marker=dict(colors=px.colors.qualitative.Prism)), 1, 3)                                                             # contact
    fig.add_trace(go.Pie(labels=filtered_df['education'].value_counts().index, values=filtered_df['education'].value_counts().values, showlegend=False,
                         hole=.4, textinfo='label+percent', textposition='inside', insidetextorientation='radial',
                         marker=dict(colors=px.colors.qualitative.G10)), 4, 3)                                                                 # education
    fig.add_trace(go.Pie(labels=filtered_df['marital'].value_counts().index, values=filtered_df['marital'].value_counts().values, showlegend=False,
                         hole=.4, textinfo='label+percent', textposition='inside', insidetextorientation='radial',
                         marker=dict(colors=px.colors.qualitative.G10[5:])), 4, 5)                                                                  # marital   

    # Box
    fig.add_trace(go.Box(x=filtered_df['age'], name='', showlegend=False, marker_color=px.colors.qualitative.G10[0] ), 1, 1)  # age
    fig.add_trace(go.Box(x=filtered_df['duration'], name='', showlegend=False, marker_color=px.colors.qualitative.Prism[6]), 1, 5)  # duration
    fig.add_trace(go.Box(y=filtered_df[(filtered_df['previous'] < 40)]["previous"], name='previous', showlegend=False, marker_color=px.colors.qualitative.Prism[8]), 2, 5)  # previous
    fig.add_trace(go.Box(y=filtered_df['campaign'], name='current', showlegend=False, marker_color=px.colors.qualitative.Prism[7]), 2, 5)  # campaign
    fig.add_trace(go.Box(x=filtered_df['balance'], name='', showlegend=False, marker_color=px.colors.qualitative.Dark2[0]), 1, 7)  # balance

    # Bar
    # loan
    fig.add_trace(go.Bar(x=filtered_df[filtered_df['loan'] == 0]["loan"].value_counts().values, y=["Personal Loan"],
                         name='No', orientation='h', marker_color=px.colors.qualitative.Dark2[1], showlegend=False), 2, 8)
    fig.add_trace(go.Bar(x=filtered_df[filtered_df['loan'] == 1]["loan"].value_counts().values, y=["Personal Loan"],
                         name='Yes', orientation='h', marker_color=px.colors.qualitative.Dark2[4], showlegend=False), 2, 8)
    # housing
    fig.add_trace(go.Bar(x=filtered_df[filtered_df['housing'] == 0]["housing"].value_counts().values, y=["Housing Loan"], orientation='h',
                         name='No', marker_color=px.colors.qualitative.Dark2[1], showlegend=False), 3, 8)
    fig.add_trace(go.Bar(x=filtered_df[filtered_df['housing'] == 1]["housing"].value_counts().values, y=["Housing Loan"], orientation='h',
                         name='Yes', marker_color=px.colors.qualitative.Dark2[4], showlegend=False), 3, 8) 
    # default
    fig.add_trace(go.Bar(x=filtered_df[filtered_df['default'] == 0]["default"].value_counts().values, y=["Credit in Default"], orientation='h',
                         name='No', marker_color=px.colors.qualitative.Dark2[1], showlegend=False), 4, 8)
    fig.add_trace(go.Bar(x=filtered_df[filtered_df['default'] == 1]["default"].value_counts().values, y=["Credit in Default"], orientation='h',
                         name='Yes', marker_color=px.colors.qualitative.Dark2[4], showlegend=False), 4, 8) 
    # poutcome
    fig.add_trace(go.Bar(x=filtered_df[filtered_df['poutcome'] == "Failure"]["poutcome"].value_counts().values, y=["Previous Term Deposit"], orientation='h',
                         name='No', marker_color=px.colors.qualitative.Dark2[1], showlegend=True), 5, 8)
    fig.add_trace(go.Bar(x=filtered_df[filtered_df['poutcome'] == "Success"]["poutcome"].value_counts().values, y=["Previous Term Deposit"], orientation='h',
                         name='Yes', marker_color=px.colors.qualitative.Dark2[4], showlegend=True), 5, 8) 
    fig.add_trace(go.Bar(x=filtered_df[filtered_df['poutcome'] == "Unknown"]["poutcome"].value_counts().values, y=["Previous Term Deposit"], orientation='h',
                         name='Unknown', marker_color='grey', showlegend=True), 5, 8)

    # Treemap
    fig.add_trace(go.Treemap(labels=filtered_df['job'].value_counts().index, values=filtered_df['job'].value_counts().values, parents=[''] * len(filtered_df['job'].unique()), 
                             marker=dict(colors=px.colors.qualitative.G10[2:])), 2, 1)  # job

    fig.update_xaxes(visible=False, row=2, col=8)  # loan
    fig.update_xaxes(visible=False, row=3, col=8)  # housing
    fig.update_xaxes(visible=False, row=4, col=8)  # default
    fig.update_xaxes(visible=False, row=5, col=8)  # poutcome
    fig.update_yaxes(visible=False, row=3, col=3)   # month

    # Update layout
    fig.update_layout(height=800, width=1400, barmode='stack', showlegend=True, template="plotly_white",
                      title={'text': "Characteristics of Subscribing clients", 'font': {'size': 25}, 'x': 0.5, 'xanchor': 'center', 'y': 0.98, 'yanchor': 'top'},
                      margin=dict(t=100),
                      shapes=[
                        # banking
                        dict(type="rect", xref="paper", yref="paper",
                            x0=0.765, x1=1, y0=0, y1=1.05, line=dict(color="green", width=2)),
                        # campaign
                        dict(type="rect", xref="paper", yref="paper",
                            x0=0.255, x1=0.755, y0=0.385, y1=1.05, line=dict(color="red", width=2)),
                        # Vertical part of the L
                        dict(type="rect", xref="paper", yref="paper",
                            x0=0.0, x1=0.245, y0=0, y1=1.05, line=dict(color="blue", width=2)),
                        # Horizontal part of the L
                        dict(type="rect", xref="paper", yref="paper",
                            x0=0.245, x1=0.755, y0=0.0, y1=0.375, line=dict(color="blue", width=2))],
                                )

    return dcc.Graph(figure=fig)

# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=10000)

