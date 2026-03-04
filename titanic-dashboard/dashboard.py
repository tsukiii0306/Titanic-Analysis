import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import KNNImputer
from sklearn import set_config
from sklearn.utils import estimator_html_repr
import streamlit.components.v1 as components

#页面设置
set_config(display='diagram') #pipeline
st.set_page_config(
    page_title="Titanic Survival Analysis",
    page_icon=":ship:",
    layout="wide",
)

# 样式设置
def set_app_style():
    sns.set_theme(style="white") 
    plt.rcParams.update({
        'font.size': 9,                  # fontsize
        'axes.titlesize': 10,            # 标题大小
        'axes.titleweight': 'normal',    # 标题粗细
        'axes.titlecolor': '#333333',    # 标题颜色
        'axes.labelcolor': '#333333',    # 轴标签颜色
        'xtick.labelsize': 9,            # X轴刻度大小
        'ytick.labelsize': 9,            # Y轴刻度大小
        'xtick.color': '#333333',        # 刻度颜色
        'ytick.color': '#333333',
        'axes.grid': True,               # 默认开启网格
        'grid.linestyle': '--',          # 虚线网格
        'grid.alpha': 0.3,               # 网格透明度
        'axes.spines.top': False,        # 全局去掉上边框
        'axes.spines.right': False,      # 全局去掉右边框
        'axes.spines.left': False,       # 全局去掉左边框
        'figure.titlesize': 12,
        'axes.titlepad':20,
        'axes.autolimit_mode':'round_numbers', # 自动取整到最近的刻度
        'axes.prop_cycle':plt.cycler(color=['#5F749D'])
    })
set_app_style()

# 自定义类
# sex
class sextransformer(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        X_copy = X.copy()
        X_copy['sex_numeric'] = X_copy['sex'].apply(lambda x:1 if x == 'female' else 0)
        return X_copy
# family size
class familysizetransformer(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        X_copy = X.copy()
        X_copy['family'] = X_copy.sibsp + X_copy.parch
        X_copy['family_size'] = X_copy.family.apply(lambda x:'alone' if x == 0 else 'small' if x <= 2 else 'large') 
        return X_copy
# age (title + fit + predict + fill in + group) （after family）
# 1. randomforest
class agetransformer(BaseEstimator,TransformerMixin):
    def __init__(self, max_depth=4, min_samples_leaf=3, random_state=3):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model = RandomForestRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state)
    def feature_process(self,X):
        X_copy = X.copy()
        X_copy['avg_fare'] = np.log1p(X_copy.fare / (X_copy.family + 1))
        X_copy['title'] = X_copy.name.apply(lambda x:x.split(',')[1].split('.')[0].strip())
        X_copy['embarked_numeric'] = X_copy.embarked.apply(lambda x:1 if x == 'C' else (2 if x == 'S' else 3))
        def get_title(title):
            if title == 'Master':
                return 1
            elif title == 'Miss':
                return 2
            elif title == 'Mr':
                return 3
            elif title == 'Mrs':
                return 4
            else:
                return 5
        X_copy['title_numeric'] = X_copy.title.apply(get_title)
        return X_copy
    def fit(self,X,y=None):
        X_processed = self.feature_process(X)
        train_df = X_processed[X_processed.age.notnull()]
        X_train = train_df[['pclass','sex_numeric','family','avg_fare','embarked_numeric','title_numeric']]
        y_train = train_df.age
        self.model.fit(X_train,y_train)
        return self
    def transform(self,X,y=None):
        X_processed = self.feature_process(X)
        test_df = X_processed[X_processed.age.isnull()]
        X_test = test_df[['pclass','sex_numeric','family','avg_fare','embarked_numeric','title_numeric']]
        predicted_age = self.model.predict(X_test) 
        X_processed['age_filled'] = X_processed.age
        age_bool = X_processed.age.isnull()
        X_processed.loc[age_bool,'age_filled'] = predicted_age 
        def get_age_group(age):
            if age <= 10:
                return 'Children'
            elif age <= 18:
                return 'Teenager'
            elif age <= 60:
                return 'Adult'
            else:
                return 'Senior'
        X_processed['age_group'] = X_processed.age_filled.apply(get_age_group)
        return X_processed
# 2. kkn
class agetransformer_kkn(BaseEstimator,TransformerMixin):
    def __init__(self, n_neighbors=5, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        self.feature = ColumnTransformer(
            [
                ('age','passthrough',['age']),
                ('sex','passthrough',['sex_numeric']),
                ('numeric',StandardScaler(),['family','avg_fare']),
                ('category',OneHotEncoder(drop='first'),['pclass','embarked_numeric','title_numeric'])
            ]
        )
    def feature_processing(self,X):
        X_copy = X.copy()
        X_copy['avg_fare'] = np.log1p(X_copy.fare / (X_copy.family + 1))
        X_copy['embarked_numeric'] = X_copy.embarked.apply(lambda x:1 if x == 'C' else 2 if x == 'S' else 3)
        X_copy['title'] = X_copy.name.apply(lambda x:x.split(',')[1].split('.')[0].strip())
        def get_title(title):
            if title == 'Master':
                return 1
            elif title == 'Miss':
                return 2
            elif title == 'Mr':
                return 3
            elif title == 'Mrs':
                return 4
            else:
                return 5
        X_copy['title_numeric'] = X_copy.title.apply(get_title)
        return X_copy
    def fit(self,X,y=None):
        X_copy = X.copy()
        X_features = self.feature_processing(X_copy)
        X_processed = self.feature.fit_transform(X_features)
        self.imputer.fit(X_processed)
        return self
    def transform(self,X,y=None):
        X_copy = X.copy()
        X_features = self.feature_processing(X_copy)
        X_processed = self.feature.transform(X_features)
        X_age_filled = self.imputer.transform(X_processed)
        X_copy['age_filled'] = X_age_filled[:,0]
        def get_age_group(age):
            if age <= 10:
                return 'Children'
            elif age <= 18:
                return 'Teenager'
            elif age <= 60:
                return 'Adult'
            else:
                return 'Senior'
        X_copy['age_group'] = X_copy.age_filled.apply(get_age_group)
        return X_copy
# fare (after age)
class faretransformer(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None): 
        df_fare = X.groupby('ticket').agg(
            children_count = ('age_filled',lambda x:(x <= 10).sum()),   
            adult_count = ('age_filled',lambda x:(x > 10).sum()),
            family_fare = ('fare',max)
        )
        df_fare['adult_fare'] = 2 * df_fare.family_fare / (df_fare.children_count + 2 * df_fare.adult_count)
        df_fare['children_fare'] = df_fare.family_fare / (df_fare.children_count + 2 * df_fare.adult_count)
        # create a mapping between ticket and fare
        self.dic_adult_fare_ = df_fare.adult_fare.to_dict()    
        self.dic_children_fare_ = df_fare.children_fare.to_dict()
        # create mapping between pclass and adult/children median fare
        X['fare_allocated'] = X.apply(lambda x:self.dic_children_fare_[x.ticket] if x.age_filled <=10 else self.dic_adult_fare_[x.ticket], axis = 1)
        self.dic_adult_median_ = X[X.age_filled > 10].groupby('pclass')['fare_allocated'].median().to_dict()
        self.dic_children_median_ = X[X.age_filled <= 10].groupby('pclass')['fare_allocated'].median().to_dict() # 报错:训练集没有抽到class1
        # calculate overall fare median as the final filling value
        self.fare_median_ = X.fare.median() 
        return self
    def transform(self,X,y=None):
        X_copy = X.copy()
        def log_fare(row):
            if row.age_filled <= 10:
                fare = self.dic_children_fare_.get(row.ticket, self.dic_children_median_.get(row.pclass, self.fare_median_))  
            else:
                fare = self.dic_adult_fare_.get(row.ticket,self.dic_adult_median_.get(row.pclass, self.fare_median_))
            return np.log1p(fare)
        X_copy['new_fare'] = X_copy.apply(log_fare, axis = 1)
        return X_copy

# sex & pclass 
class sexpclasstransformer(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        X_copy = X.copy()
        X_copy['sex_pclass'] = X_copy.sex.apply(lambda x:x.strip()) + X_copy.pclass.apply(lambda x:str(x).strip())
        return X_copy

# columntransformer
ct = ColumnTransformer(
    [
        ('scaler',StandardScaler(),['new_fare']),
        ('encoding',OneHotEncoder(drop='first'),['pclass','sex_numeric','family_size','age_group','embarked','sex_pclass'])
    ]
)

# family
class familytransformer(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        X_copy = X.copy()
        X_copy['family'] = X_copy.sibsp + X_copy.parch
        return X_copy

# age
class knn_agetransformer(BaseEstimator,TransformerMixin):
    def __init__(self,n_neighbors = 5,weights = 'uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.imputer = KNNImputer(n_neighbors=n_neighbors,weights=weights)
        self.feature = ColumnTransformer(
            [
                ('age','passthrough',['age']),
                ('sex','passthrough',['sex_numeric']),
                ('numeric',StandardScaler(),['family','avg_fare']),
                ('category',OneHotEncoder(drop='first'),['pclass','embarked_numeric','title_numeric'])
            ]
        )
    def feature_processing(self,X):
        X_copy = X.copy()
        X_copy['avg_fare'] = np.log1p(X_copy.fare / (X_copy.family + 1))
        X_copy['embarked_numeric'] = X_copy.embarked.apply(lambda x:1 if x == 'C' else 2 if x == 'S' else 3)
        X_copy['title'] = X_copy.name.apply(lambda x:x.split(',')[1].split('.')[0].strip())
        def get_title(title):
            if title == 'Master':
                return 1
            elif title == 'Miss':
                return 2
            elif title == 'Mr':
                return 3
            elif title == 'Mrs':
                return 4
            else:
                return 5
        X_copy['title_numeric'] = X_copy.title.apply(get_title)
        return X_copy
    def fit(self,X,y=None):
        X_copy = X.copy()
        X_features = self.feature_processing(X_copy)
        X_processed = self.feature.fit_transform(X_features)
        self.imputer.fit(X_processed)
        return self
    def transform(self,X,y=None):
        X_copy = X.copy()
        X_features = self.feature_processing(X_copy)
        X_processed = self.feature.transform(X_features)
        X_age_filled = self.imputer.transform(X_processed)
        X_copy['age_filled'] = X_age_filled[:,0]
        return X_copy

ct2 = ColumnTransformer(
    [
        ('encoding',OneHotEncoder(),['embarked','pclass']),
        ('features','passthrough',['age_filled','sex_numeric','fare','family'])
    ]
)


@st.cache_data
def load_model_results():
    return joblib.load('titanic-dashboard/model_analysis_results.pkl')

model_results = load_model_results()


# 数据处理
@st.cache_data
def load_data():
    df = pd.read_csv('titanic-dashboard/train.csv')
    df.columns = df.columns.str.lower()
    return df
df = load_data()

def get_data(df, age_method, fare_allocation, fare_transform):
    df_processed = df.copy()

    # age 
    df_processed['title'] = df_processed.name.apply(lambda x:x.split(',')[1].split('.')[0].strip())
    if age_method == 'Median by Title':
        df_processed['age_filled'] = df_processed.groupby('title')['age'].transform(lambda x:x.fillna(x.median()))
    elif age_method == 'Mean by Title':
        df_processed['age_filled'] = df_processed.groupby('title')['age'].transform(lambda x:x.fillna(x.mean()))
    else:
        df_processed = df_processed.dropna(subset = ['age']) # 只删除age空缺的样本
        df_processed['age_filled'] = df_processed['age'] #注意下面分组的数据来源
    def age_grouping(age):
        if age <= 10:
            return 'Children'
        elif age <= 18:
            return 'Teenager'
        elif age <= 60:
            return 'Adult'
        else:
            return 'Senior'
    df_processed['age_group'] = df_processed.age_filled.apply(age_grouping)

    # fill in port
    df_processed['port'] = df_processed['embarked']
    df_processed['port'] = df_processed['port'].fillna('C')

    # family group
    df_processed['family'] = df_processed.sibsp + df_processed.parch
    df_processed['family_size'] = df_processed.family.apply(lambda x:'Alone' if x == 0 else 'Small Size' if x <= 2 else 'Large Size')

    # fare
    # 1.allocation
    df_processed['children'] = df_processed.age_filled.apply(lambda x:1 if x <= 10 else 0)
    df_family_status = df_processed.groupby('ticket').agg(
    family_counts = ('ticket','count'),
    children_count = ('children','sum'),
    fare_price = ('fare','max')
    )
    if fare_allocation == 'Direct average':
        df_family_status['avg_price'] = df_family_status.fare_price / df_family_status.family_counts
        dic_avg_price = df_family_status.avg_price.to_dict()
        df_processed['fare_new'] = df_processed.apply(lambda x:dic_avg_price[x['ticket']], axis = 1)
    elif fare_allocation == 'Half-price child':
        df_family_status['adult_count'] = df_family_status.family_counts - df_family_status.children_count
        df_family_status['adult_price'] = 2 * df_family_status.fare_price / (df_family_status.children_count + 2 * df_family_status.adult_count)
        df_family_status['children_price'] = df_family_status.fare_price / (df_family_status.children_count + 2 * df_family_status.adult_count)
        dic_children = df_family_status.children_price.to_dict()
        dic_adult = df_family_status.adult_price.to_dict()
        df_processed['fare_new'] = df_processed.apply(lambda x:dic_children[x['ticket']] if x.children == 1 else dic_adult[x['ticket']], axis = 1)
    else:
        df_processed['fare_new'] = df['fare']
    # 2.transform
    if fare_transform == 'Log':
        df_processed['fare_transformed'] = np.log1p(df_processed.fare_new)
    elif fare_transform == 'Square':
        df_processed['fare_transformed'] = df_processed.fare_new ** 2
    elif fare_transform == 'Exp':
        df_processed['fare_transformed'] = np.exp(df_processed.fare_new)
    else:
        df_processed['fare_transformed'] = df_processed['fare_new']

    return df_processed

# 侧栏
with st.sidebar:
    # 1. EDA数据处理
    with st.expander("🛠️ Data Preprocessing", expanded=False):
        st.subheader('Missing Value')
        age_method = st.radio('Method for Age:', ['Median by Title', 'Mean by Title', 'Drop'])
        
        st.subheader('Group Ticket')
        fare_allocation = st.radio('Method for Group Fares:', ['Keep original', 'Direct average', 'Half-price child'])

        st.subheader('Fare Transform')
        fare_transform = st.selectbox("Select Transformation:", ["Original", "Log"])

    # 2. 模型选择
    with st.expander("🤖 Model Selection", expanded=True):
        model_type = st.selectbox("Select Model:", ['Logistic Regression', 'Random Forest'])
        age_impute_method = st.radio('Choose Age Imputation Method:',['Random Forest', 'KNN'])
        selected_method = age_impute_method

current_df = get_data(df, age_method, fare_allocation, fare_transform)

# 标题 + overview
st.title('Titanic Survival Factor Exploration')
st.markdown("""
This dashboard uses data visualization to analyze the key factors that affected passenger survival during the 1912 Titanic disaster.
""")
st.subheader('Overview')
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric('Total Passengers', len(current_df))
with col2:
    survivors_count = current_df['survived'].sum()
    st.metric('Survivors', survivors_count)
with col3:
    died_count = len(current_df) - survivors_count
    st.metric('Fatalities', died_count)
with col4:
    survival_rate = (survivors_count / len(current_df)) * 100
    st.metric('Total Survival Rate', f'{survival_rate:.2f}%')

# 单因素分析
st.subheader('Single Factor Exploration')
feature_map = {
    'Sex': 'sex',
    'Pclass': 'pclass',
    'Family Size': 'family_size',
    'Port': 'port',
    'Age': 'age_group',
    'Fare': 'fare_transformed'
}
selected_label = st.selectbox('Choose a factor',['Sex', 'Pclass', 'Family Size', 'Port', 'Age', 'Fare'])
factor = feature_map[selected_label]
if selected_label == 'Age':
    col1, col2, col3 = st.columns(3)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        sns.histplot(data=current_df, x='age_filled', bins=30, ax=ax1)
        ax1.set(title='Age Distribution', xlabel='Age', ylabel='Count')
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.countplot(data=current_df, x='age_group', ax=ax2)
        ax2.set(title='Age Group Distribution', xlabel='Age Group', ylabel='Count')
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%d', padding=6)
        st.pyplot(fig2)
    with col3:
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        sns.barplot(data=current_df, x='age_group', y='survived', errorbar=None, ax=ax3)
        ax3.set(title='Survival Rate by Age Group', xlabel='Age Group', ylabel='Survival Rate (%)')
        for container in ax3.containers:
            ax3.bar_label(container, fmt=lambda x: f"{x*100:.1f}%", padding=6)
        st.pyplot(fig3)
elif selected_label == 'Fare':
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        sns.histplot(data=current_df, x='fare_transformed', bins=20, ax=ax1)
        ax1.set(title='Fare Distribution', xlabel='Fare', ylabel='Count')
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        sns.violinplot(data=current_df, x='survived', y='fare_transformed', ax=ax2)
        ax2.set(title='Fare Distribution by Survival Status', xlabel='Survived Status', ylabel='Fare')
        st.pyplot(fig2)
else:
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        sns.countplot(data=current_df, x=factor, ax=ax1)
        ax1.set(title=f'{selected_label} Distribution', xlabel=selected_label, ylabel='Count')
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%d', padding=6)
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=current_df, x=factor, y='survived', errorbar=None, ax=ax2)
        ax2.set(title=f'Survival Rate by {selected_label}', xlabel=selected_label, ylabel='Survival Rate (%)')
        for container in ax2.containers:
            ax2.bar_label(container, fmt=lambda x: f"{x*100:.1f}%", padding=6)
        st.pyplot(fig2)
# 动态说明
analysis = {
    'sex':{
        'Observations':'* Although the number of male passengers was almost twice higher than females, female passengers accounted for the majority of survivors, and the survival rate of females (74.20%) was significantly higher than males (18.90%). This indicates that gender is a strong feature to the model.',
        'Reasons':'* The reason might be the women and children first principle. In the age analysis, we can observe that children also have higher survival rate.'
    },
    'pclass':{
        'Observations':'* Class 3 had the highest number of passengers but the lowest survival rate. There is a positive correlation between Pclass and survival rate: The higher the class, the higher the survival rate.',
        'Reasons':"""* **Priority:** Passengers in Class 1 may have had privileges during evacuation; they had the priority to resources. Moreover, the crew may have put the safety of Class 1 before other classes, such as providing them with better guidance.
        * **Location:** First class cabins were usually located on the upper decks, which allowed those passengers to quickly reach life-saving resources and follow simpler escape routes."""
    },
    'family_size':{
        'Explanations':"* **Alone**: 1 person (Traveling without family)\n* **Small Size**: 1 - 2 people (excluding the passenger)\n* **Large Size**: 3 or more people",
        'Observations':'* Passengers accompanied with 1 - 2 family members has a significant higher survival rate than passengers who traveled alone or with more than 3 companions.',
        'Reasons':"""* **Mutual assistance:** improve survival chances among family with small size.
        * **Low efficiency:** may struggle to ensure everyone's safety among family with large size."""
    },
    'age_group':{
        'Explanations':"* **Children**: [0, 10]\n* **Teenager**: (10,18]\n* **Adult**: (18,60]\n* **Senior**: (60,)",
        'Observations':"* The bar chart shows a decline in survival rate as age increases and the survival rate of children(almost 60%) is nearly double of the senior group. What's more, as the adult passengers account for the majority of the samples, its survival rate(38.88%) is close to the overall(38.38%).",
        'Reasons':'* Women and children first principle'
    },
    'port':{
        'Observations':'* Port C with the highest proportion of class 1 also has the highest survival rate, but when it comes to port Q and port S, the distribution of pclass can not explain why port Q with almost all passengers from class 3 has a slight higer survival rate than port S. Therefore, there might be some other factors result in the contradiction.'
    },
    'fare_transformed':{
        'Observations':'* The median fare of survivors is significantly higher than the died group. In the death group, there is a significantly wider distribution in the low fare price, and then drop rapidly when the price rise up.'
    }
}
current_analysis = analysis.get(factor)
info_text = ""
if current_analysis.get('Explanations'):
    info_text += f"**Explanations:**\n{current_analysis['Explanations']}\n\n"
if current_analysis.get('Observations'):
    info_text += f"**Observations:**\n{current_analysis['Observations']}\n\n"

if current_analysis.get('Reasons'):
    info_text += f"**Reasons:**\n{current_analysis['Reasons']}\n\n"
st.info(info_text)
st.markdown("---") 

# 多因素分析
st.subheader('Multiple Factor Exploration')
st.markdown("")
with st.expander('View Analysis Premises', expanded=False):
    st.markdown("""
    **All multi-factor analyses on this page are based on the following data preprocessing methods:**
    1. Missing Age Values: median age by title
    2. Group Fare Calculation: Child tickets are priced at half price
    3. Fare Transformation: log transformation
    """)
st.markdown("#### Exploration I: Passenger Class and Gender Effects")
with st.expander("Observation I: Fare Contributes Little to Survival Within the Same Passenger Class", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        sns.violinplot(data=current_df, x='pclass', y='fare_transformed', ax=ax1)
        ax1.set(title='Fare Distribution by Pclass', xlabel='Pclass', ylabel='Fare')
        st.pyplot(fig1)
    with col2:
        st.info("""
    **Observations:**       
    * **Pclass 1:** The main body of the fare distribution for survivors and non-survivors is very similar, with their medians concentrate on the same price point. However, the survivor group has a few extremely high fare values at the upper, which might suggest that the extreme high fare means highest survival priority. 
              
    * **Pclass 2:** The two distributions are now very close in shape, except for a small number of extremely low fare outliers in the non-survivor group.
                
    * **Pclass 3:** The two distributions are almost identical in both shape and spread which prove that fare almost has no meaningful impact on survival at all.
            
    * **Overall:** Fare and passenger class are strongly correlated. First class passengers have a significantly higher median fare than second and third class. However, when fixing the passenger class and examine the effect of fare on survival rate, it shows that fare variations have little impact. The effect of fare is mainly determined by Pclass.
    """)
    col1, col2, col3 = st.columns(3)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        sns.violinplot(data=current_df[current_df.pclass == 1], x='survived', y='fare_transformed', ax=ax1)
        ax1.set(title='Survival Effects of Fare in Pclass 1', xlabel='Survived Status', ylabel='Fare')
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.violinplot(data=current_df[current_df.pclass == 2], x='survived', y='fare_transformed', ax=ax2)
        ax2.set(title='Survival Effects of Fare in Pclass 2', xlabel='Survived Status', ylabel='Fare')
        st.pyplot(fig2)
    with col3:
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        sns.violinplot(data=current_df[current_df.pclass == 3], x='survived', y='fare_transformed', ax=ax3)
        ax3.set(title='Survival Effects of Fare in Pclass 3', xlabel='Survived Status', ylabel='Fare')
        st.pyplot(fig3)

with st.expander('Observation Ⅱ: Gender Emerges as the Dominant Survival Factor', expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        sns.countplot(data=current_df, x='sex', hue='pclass', palette='dark:#5F749D', ax=ax1)
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%d', padding=6)
        ax1.set(title='Sex Distribution by Pclass', xlabel='Sex', ylabel='Count')
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=current_df, x='sex', y='survived', hue='pclass', palette='dark:#5F749D', errorbar=None, ax=ax2)
        for container in ax2.containers:
            ax2.bar_label(container, fmt=lambda x: f"{x*100:.1f}%", padding=6)
        ax2.set(title='Survival Rate by Gender and Pclass', xlabel='Sex', ylabel='Survival Rate (%)')
        st.pyplot(fig2)
    st.info("""
    **Observations:**
    * The survival rate of female in both class 1 and 2 is larger than 90% and even in class 3, females had a significantly higher chance to survive than males, and it seems that gender is a stronger impact than class. 
    """)
st.markdown("")

st.markdown("#### Exploration Ⅱ: Multi-Factor Analysis of Port Survival")
with st.expander("Observation: Port & Pclass", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        sns.countplot(data=current_df, x='port',hue='pclass', palette='dark:#5F749D', ax=ax1)
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%d', padding=6)
        ax1.set(title='Port Distribution by Pclass', xlabel='Port', ylabel='Count')
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=current_df, x='port', y='survived', hue='pclass', palette='dark:#5F749D', errorbar=None, ax=ax2)
        for container in ax2.containers:
            ax2.bar_label(container, fmt=lambda x: f"{x*100:.1f}%", padding=6)
        ax2.set(title='Survival Rate by Port and Pclass', xlabel='Port', ylabel='Survival Rate (%)')
        st.pyplot(fig2)
    st.info("""
    **Observations:**
    * As Port S with the largest number of 3rd class passengers, it had the lowest survival rate. However when we focus on the 3rd class, the survival rate for 3rd class passengers at Port S is significantly lower than both Port C and Port Q. The observation suggests that class alone cannot fully explain the lower survival rate at Port S.
    * Port S has passengers in 1st and 2nd class, while Port Q is almost 3rd class. However Port Q has higher survival rate than Port S.
    """)

with st.expander('Assumption I: Port & Sex'):
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        current_df['sex_numeric'] = current_df.sex.apply(lambda x:1 if x == 'female' else 0)
        df_female_proportion = current_df.groupby('port')[['sex_numeric']].mean() # series to dataframe
        sns.barplot(data=df_female_proportion, x='port', y='sex_numeric', ax=ax1)
        for container in ax1.containers:
            ax1.bar_label(container, fmt=lambda x: f"{x*100:.1f}%", padding=6)
        ax1.set(title='Female Proportion by Port', xlabel='Port', ylabel='Proportion (%)')
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=current_df, x='port', y='survived', hue='sex', palette='dark:#5F749D', errorbar=None, ax=ax2)
        for container in ax2.containers:
            ax2.bar_label(container, fmt=lambda x: f"{x*100:.1f}%", padding=6)
        ax2.set(title='Survival Rate by Gender and Port', xlabel='Sex', ylabel='Survival Rate (%)')
        st.pyplot(fig2)
    st.info("""
    **Observations:**
    * Port S has a notably lower female proportion compared to Port C and Port Q which can explain the lower survival rate in Port S.
    * Although Port S has a lower female proportion, there is a contradiction: female passengers from Port Q (almost from 3rd class) have a higher survival rate than females from Port S (a mix of all passenger classes). This directly challenges the assumption that female from higher class has higher survival rate.
    """)

with st.expander('Assumption Ⅱ: Port & Sex & Age'):
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        sns.violinplot(data=current_df, x='port', y='age_filled', ax=ax1)
        for container in ax1.containers:
            ax1.bar_label(container, fmt=lambda x: f"{x*100:.1f}%", padding=6)
        ax1.set(title='Age Distribution by Port', xlabel='Port', ylabel='Age')
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        sns.violinplot(data=current_df[current_df.sex == 'female'], x='port', y='age_filled', ax=ax2)
        for container in ax2.containers:
            ax2.bar_label(container, fmt=lambda x: f"{x*100:.1f}%", padding=6)
        ax2.set(title='Age Distribution of Female Passengers by Port', xlabel='Port', ylabel='Female Age')
        st.pyplot(fig2)
    st.info("""
    **Observations:**
    * Female passengers in port Q is significantly lower than other ports, it might be the reason why female in port Q has higher survival rate. However, the sample of female in port Q is small(36). 
            
    **Next step:**
    * Find the relationship between female survival rate within different age to examine whether younger women has higher survival rate.
    """)
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        sns.histplot(data=current_df[current_df.sex == 'female'], x='age_filled', ax=ax1)
        ax1.set(title='Distribution of Female Age', xlabel='Age', ylabel='Count')
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        sns.violinplot(data=current_df[current_df.sex == 'female'], x='survived', y='age_filled', ax=ax2)
        ax2.set(title='Distribution of Female Survival Rate by Age', xlabel='Survived Status', ylabel='Female Age')
        st.pyplot(fig2)
    st.info("""
    **Observations:**
    * The age advantage seems to be a specific factor in port Q, the overall distribution shows that most of the survived females are older than those died. 
    """)

with st.expander('Assumption Ⅲ: Port & Sex & Age & Family Size'):
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        sns.countplot(data=current_df[current_df.sex == 'female'], x='port', hue='family_size', palette='dark:#5F749D', ax=ax1)
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%d', padding=6)
        ax1.set(title='The Distribution of Females in Different Family Size by Port', xlabel='Port', ylabel='Count')
        ax1.legend(title="family Size", loc='upper right', fontsize=8)
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=current_df[current_df.sex == 'female'], x='port', y='survived', hue='family_size', palette='dark:#5F749D', errorbar=None, ax=ax2)
        for container in ax2.containers:
            ax2.bar_label(container, fmt=lambda x: f"{x*100:.1f}%", padding=8)
        ax2.set(title='Survival Rate of Females in Different Family Size by Port', xlabel='Survived Status', ylabel='Survival Rate (%)')
        ax2.legend(title="family Size", loc='lower right', fontsize=8)
        st.pyplot(fig2)
    st.info("""
    **Observations:**
    * The reason for the higher survival rate for female in port Q seems related with the family size, it shows that almost no female from large family size in port Q while port S have the highest number of females from large family. 
    """)
with st.expander('Conclusions'):
    st.info("""
    * The lowest survival rate at Port S is resulted from a combination of lower proportion of females, more 3rd class passengers and higher proportion of large families.
    * It seems that higher survival rate for female in Port S is connected with family size, but the sample is very small and there might be some other reasons.
    """)

st.markdown('#### Exploration Ⅲ: Gender and Family Size Interaction Effect')
with st.expander("Observation I: Stratified Analysis by Gender", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        sns.countplot(data=current_df, x='sex',hue='family_size', palette='dark:#5F749D', ax=ax1)
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%d', padding=6)
        ax1.set(title='Sex Distribution by Family Size', xlabel='Sex', ylabel='Count')
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=current_df, x='sex', y='survived', hue='family_size', palette='dark:#5F749D', errorbar=None, ax=ax2)
        for container in ax2.containers:
            ax2.bar_label(container, fmt=lambda x: f"{x*100:.1f}%", padding=6)
        ax2.set(title='Survival Rate by Sex and Family Size', xlabel='Sex', ylabel='Survival Rate (%)')
        st.pyplot(fig2)
with st.expander("Observation Ⅱ: Stratified Analysis by Family Size", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        sns.countplot(data=current_df, x='family_size',hue='sex', palette='dark:#5F749D', ax=ax1)
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%d', padding=6)
        ax1.set(title='Family Size Distribution by Sex', xlabel='Family Szie', ylabel='Count')
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=current_df, x='family_size', y='survived', hue='sex', palette='dark:#5F749D', errorbar=None, ax=ax2)
        for container in ax2.containers:
            ax2.bar_label(container, fmt=lambda x: f"{x*100:.1f}%", padding=6)
        ax2.set(title='Survival Rate by Sex and Family Size', xlabel='Family Szie', ylabel='Survival Rate (%)')
        st.pyplot(fig2)
with st.expander("Conclusions", expanded=False):
    st.info("""
    **For female:**
    * The survival advantage of female seems to come from gender itself, and it doesn't seem to be affected much by whether the female traveled alone or with small family members.
    * However, females from large size family have significant lower survival rate, which is consistent with the observation in port S(lower survival rate for females).
    **For male:**
    * For passengers traveling alone, the huge proportion of males leads to a lower overall survival rate. 
    * In small families, the male to female ratio is roughly equal, and the survival rate of men indeed increased a lot and it might be related with the higher survival rates of the women around them.
    **Explanation:**Negative coefficient of small size family
    * The female samples in small family is mainly explained by the feature sex while the rest survival rate of male is below the average level, so the model give it a negative coefficient.
    """)   
st.markdown("---")   

# EDA分析结论
st.subheader("Conclusion")
st.info("""
    * In conclusion, **Gender** was the **most important factor** for survival, followed by **Passenger Class**. **Age** protected the children, and **Family Size** provided a survival advantage for small groups. Finally **Fare** was highly correlated with class but had almost no impact on survival within the same passenger class.
    """)
st.markdown("---")

# Model selection
if model_type == 'Logistic Regression' and selected_method:
    st.subheader(f'Model Selection: {model_type}')
    st.markdown("")
    st.markdown("")
    current_result = model_results[model_type][selected_method]
    current_grid = current_result['grid']
    current_train_accuracy = current_result['test']
    # key metrics
    col1, col2, col3 = st.columns(3)
    col1.metric('Best CV Score', f'{current_grid.best_score_ * 100:.2f}%')
    col2.metric('Train set Accuracy',f'{current_train_accuracy * 100:.2f}%')
    col3.metric('Age Imputation',selected_method)
   
    # pipeline
    st.markdown("")
    st.markdown('#### Pipeline Architecture')
    coll, col2, col3 = st.columns([1, 8, 1])
    with col2:
        with st.container(border=True):
            best_pipeline = current_grid.best_estimator_
            raw_html = estimator_html_repr(best_pipeline)
            centered_html = f"""  
        <div style="text-align: center; width: 100%; padding-top: 0;">
            <div style="display: inline-block; text-align: left; margin: 0 auto;">
                {raw_html}
            </div>
        </div>
        """  # 居中显示
            components.html(centered_html, height=450, scrolling=True)

    # parameters
    st.markdown("")
    st.markdown("#### Best Hyperparameters")
    st.markdown("")
    params_df = pd.DataFrame.from_dict(
        current_grid.best_params_, 
        orient='index', 
        columns=['Value']
    ).rename_axis('Parameters')
    params_df['Value'] = params_df['Value'].astype(str)
    st.dataframe(params_df, width=400, height=140)

elif model_type == 'Random Forest' and selected_method:
    st.subheader(f'Model Selection: {model_type}')
    st.markdown("")
    st.markdown("")
    current_result = model_results[model_type][selected_method]
    current_grid = current_result['grid']
    current_train_accuracy = current_result['test']
    # key metrics
    col1, col2, col3 = st.columns(3)
    col1.metric('Best CV Score', f'{current_grid.best_score_ * 100:.2f}%')
    col2.metric('Train set Accuracy',f'{current_train_accuracy * 100:.2f}%')
    col3.metric('Age Imputation',selected_method)
   
    # pipeline
    st.markdown("")
    st.markdown('#### Pipeline Architecture')
    coll, col2, col3 = st.columns([1, 8, 1])
    with col2:
        with st.container(border=True):
            best_pipeline = current_grid.best_estimator_
            raw_html = estimator_html_repr(best_pipeline)
            centered_html = f"""  
        <div style="text-align: center; width: 100%; padding-top: 0;">
            <div style="display: inline-block; text-align: left; margin: 0 auto;">
                {raw_html}
            </div>
        </div>
        """  # 居中显示
            components.html(centered_html, height=450, scrolling=True)

    # parameters
    st.markdown("")
    st.markdown("#### Best Hyperparameters")
    st.markdown("")
    params_df = pd.DataFrame.from_dict(
        current_grid.best_params_, 
        orient='index', 
        columns=['Value']
    ).rename_axis('Parameters')
    params_df['Value'] = params_df['Value'].astype(str)
    st.dataframe(params_df, width=400, height=140)

