import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px

# Header
st.title("Activity #4 BD & #3 CC - Streamlit EDA")
st.header("Titanic Dataset - Arpan Ghosh")
st.text("Data example:")

# Get Adult data 
@st.cache
def get_data():
    URL = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    return pd.read_csv(URL, header = 0)

df = get_data()
st.dataframe(df.head())

st.text("Code of how to get adult data:")
st.code("""
@st.cache
def get_data():
    URL = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    return pd.read_csv(URL, header = 0)

df = get_data()
st.dataframe(df.head())
""", language="python")

# 1st Interactive EDA
st.subheader("1st: Select the columns of the information you want")
default_cols = ["Name","Age","Sex", "Survived"]
cols = st.multiselect("Columns selected:", df.columns.tolist(), default=default_cols)
st.dataframe(df[cols])

st.text("Code:")
st.code("""
default_cols = ["Name","Age","Sex", "Survived"]
cols = st.multiselect("Columns selected:", df.columns.tolist(), default=default_cols)
st.dataframe(df[cols])
""", language="python")

# 2nd Interactive EDA
st.subheader("2nd: Histogram of ages")
values = st.sidebar.slider("Age", float(df.Age.min()), float(df.Age.max()), (0.42, 80.))
hist = px.histogram(df.query(f"Age.between{values}", engine="python" ), x="Age", nbins=50, title="Ages Distribution:")
hist.update_xaxes(title="Age Range")
hist.update_yaxes(title="# of people according to age")
st.plotly_chart(hist)

st.text("Code:")
st.code("""
values = st.sidebar.slider("Age", float(df.Age.min()), float(df.Age.max()), (0.42, 80.))
hist = px.histogram(df.query(f"Age.between{values}", engine="python" ), x="Age", nbins=50, title="Ages Distribution:")
hist.update_xaxes(title="Age Range")
hist.update_yaxes(title="# of people according to age")
st.plotly_chart(hist)
""", language="python")

# 3rd Interactive EDA
st.subheader("3rd: Who pays more? ")
st.text("Average fare according to sex:")
dx = df.groupby("Sex").Fare.mean().reset_index().sort_values("Fare", ascending=False)
dx.columns = ["Sex","Average Fare"]
st.table(dx)

st.text("Code:")
st.code("""
dx = df.groupby("Sex").Fare.mean().reset_index().sort_values("Fare", ascending=False)
dx.columns = ["Sex","Average Fare"]
st.table(dx)
""", language="python")

# 4th Interactive EDA
st.subheader("4th: Percentage of survivors according to their title")
st.text("Count and percentage of survivors according to the title:")
name = df["Name"].str.split(".", expand=True)
name.columns = ["Title","Name2","drop"]
df = pd.concat([df,name], axis=1)
dx = df.groupby("Title").Survived.mean().multiply(100).reset_index()
dy = df.groupby("Title").Sex.count().reset_index()
dp = pd.concat([dx,dy["Sex"]], axis=1) 
dp.columns = ["Title","% Of Survivors","Total with that title"]
st.dataframe(dp.sort_values("% Of Survivors", ascending=False))

st.text("Code:")
st.code("""
name = df["Name"].str.split(".", expand=True)
name.columns = ["Title","Name2","drop"]
df = pd.concat([df,name], axis=1)
dx = df.groupby("Title").Survived.mean().multiply(100).reset_index()
dy = df.groupby("Title").Sex.count().reset_index()
dp = pd.concat([dx,dy["Sex"]], axis=1) 
dp.columns = ["Title","% Of Survivors","Total with that title"]
st.dataframe(dp.sort_values("% Of Survivors", ascending=False))
""", language="python")

# 5th Interactive EDA
st.subheader("5th: Filtered according to their title")
Title = st.radio("Title options:", df.Title.unique())
@st.cache
def get_info(Title):
    return df.query("Title==@Title")

dx = get_info(Title)
st.dataframe(dx[["Name2","Sex","Age","Fare","Survived"]])

st.text("Code:")
st.code("""
Title = st.radio("Title options:", df.Title.unique())
@st.cache
def get_info(Title):
    return df.query("Title==@Title")

dx = get_info(Title)
st.dataframe(dx[["Name2","Sex","Age","Fare","Survived"]])
""", language="python")

agree = st.button("Do you want to see another graphics and testing?")
if agree:
    st.text("Convert a row to .json:")
    st.json({
        "Name": df['Name2'][0],
        "Sex": df['Sex'][0],
        "Age": df['Age'][0],
        "Fare": df['Fare'][0],
        "Pclass": df['Pclass'][0],
        "Survived": df['Survived'][0]
    })

    st.text("DataSet info")
    st.table(df.describe())

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.text("Total Fare by Pclass")
    sns.barplot(x='Pclass', y='Fare', data=df)
    st.pyplot()
