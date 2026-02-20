import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data processing
@st.cache_data  
def get_data():
    # load data
    df = pd.read_csv('train.csv')
    df.columns = df.columns.str.lower()
    # fill in port
    df['port'] = df['embarked']
    df['port'] = df.port.fillna('C')
    # fill in age + age group
    df['title'] = df.name.apply(lambda x:x.split(',')[1].split('.')[0].strip())
    df['age_filled'] = df.groupby('title')['age'].transform(lambda x:x.fillna(x.median()))
    def age_grouping(age):
        if age <= 10:
            return 'Children'
        elif age <= 18:
            return 'Teenager'
        elif age <= 60:
            return 'Adult'
        else:
            return 'Senior'
    df['age_group'] = df.age_filled.apply(age_grouping)
    # family group
    df['family'] = df.sibsp + df.parch
    df['family_size'] = df.family.apply(lambda x:'Alone' if x == 0 else ('Small Size' if x <=2 else 'Large Size'))
    # fare allocation
    df['children'] = df.age_filled.apply(lambda x:1 if x <= 10 else 0)
    df_family_status = df.groupby('ticket').agg(
    family_counts = ('ticket','count'),
    children_count = ('children','sum'),
    fare_price = ('fare',max)
    )
    df_family_status['adult_count'] = df_family_status.family_counts - df_family_status.children_count
    df_family_status['adult_price'] = 2 * df_family_status.fare_price / (df_family_status.children_count + 2 * df_family_status.adult_count)
    df_family_status['children_price'] = df_family_status.fare_price / (df_family_status.children_count + 2 * df_family_status.adult_count)
    dic_adult_price = df_family_status.adult_price.to_dict()
    dic_children_price = df_family_status.children_price.to_dict()
    df['fare_new'] = df.apply(lambda x:dic_children_price[x['ticket']] if x['children'] == 1 else dic_adult_price[x['ticket']],axis=1)
    df['log_fare3'] = df.fare_new.apply(np.log1p)
    return df

df = get_data()

# 样式
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
        'figure.titlesize': 12
    })
# 执行
set_app_style()

st.set_page_config(
    page_title="Titanic Survival Analysis",
    page_icon=":ship:",
    layout="wide",
)

st.title("Titanic Survival Factor Exploration")
st.markdown("""
This dashboard uses data visualization to analyze the key factors that affected passenger survival during the 1912 Titanic disaster.
""")

# overview
st.subheader("Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Passengers", len(df))

with col2:
    survived_count = df['survived'].sum()
    st.metric("Survivors", survived_count)

with col3:
    died_count = len(df) - survived_count
    st.metric("Fatalities", died_count)

with col4:
    survival_rate = (survived_count / len(df)) * 100
    st.metric("Total Survival Rate", f"{survival_rate:.2f}%")

# Exploration of Single Factors
st.subheader("Single Factor Exploration")
feature_map = {
    "Sex": "sex",
    "Pclass": "pclass",
    "Family Size": "family_size", 
    "Port": "port",
    "Age Group": "age_group",
    "Fare": "log_fare3"
}
selected_label = st.selectbox("Choose a Factor", list(feature_map.keys()))
field = feature_map[selected_label]

col_dist, col_surv = st.columns(2)

if field == 'log_fare3':
    col_dist, col_surv = st.columns(2)
    
    with col_dist:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        sns.histplot(data=df, x=field, bins=30, color='#66c2a5', ax=ax1)
        
        ax1.set_title("Fare Distribution")
        ax1.set_xlabel('Fare')
        ax1.set_ylabel("Frequency")
        st.pyplot(fig1)

    with col_surv:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        sns.violinplot(data=df, x='survived', y=field, palette="viridis", ax=ax2)
        
        ax2.set_title("Fare Distribution by Survival Status")
        ax2.set_xlabel("Survived Status(0=No, 1=Yes)")
        ax2.set_ylabel("Fare")
        st.pyplot(fig2)
else:
    with col_dist:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        sns.countplot(data=df, x=field, color='#66c2a5', ax=ax1)
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%d', padding=8)

        ax1.set_title(f"{field.capitalize()} Distribution", pad=20)
        ax1.set_xlabel(field)
        ax1.set_ylabel("Count")
        ax1.set_ylim(0, df[field].value_counts().max() * 1.25)
        st.pyplot(fig1)

    with col_surv:
        surv_data = df.groupby(field)['survived'].mean().reset_index() #把分组结果从 Series 转为 DataFrame
        surv_data['survived_pct'] = surv_data['survived'] * 100
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=surv_data, x=field, y='survived_pct', palette="viridis", ax=ax2)
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%.1f%%', padding=8)

        ax2.set_title(f"Survival Rate by {field.capitalize()}", pad=20)
        ax2.set_xlabel(field)
        ax2.set_ylabel("Survival Rate (%)")
        ax2.set_ylim(0, 115) 
        st.pyplot(fig2)

# Analysis
if field == 'sex':
    with st.container():
        st.info("""
        **Observations:** 
        * Although the number of male passengers was almost twice higher than females, female passengers accounted for the majority of survivors, and the survival rate of females (74.20%) was significantly higher than males (18.90%). This indicates that gender is a strong feature to the model.
        
        **Reasons:** 
        * The reason might be the 'women and children first' principle. In the age analysis, we can observe that children also have higher survival rate.
        """)
        
elif field == 'pclass':
    with st.container():
        st.info("""
        **Observations:**
        * Class 3 had the highest number of passengers but the lowest survival rate. There is a positive correlation between Pclass and survival rate: The higher the class, the higher the survival rate.

        **Reasons:**
        * **Priority:** Passengers in Class 1 may have had privileges during evacuation; they had the priority to resources. Moreover, the crew may have put the safety of Class 1 before other classes, such as providing them with better guidance.
        * **Location:** First class cabins were usually located on the upper decks, which allowed those passengers to quickly reach life-saving resources and follow simpler escape routes.
        """)

elif field == 'family_size':
    with st.container():
        st.info("""
        **Explanations:**
        * **Alone**: 1 person (Traveling without family)
        * **Small Size**: 1 - 2 people (excluding the passenger)
        * **Large Size**: 3 or more people
                      
        **Observations:**
        * Passengers accompanied with 1 - 2 family members has a significant higher survival rate than passengers who traveled alone or with more than 3 companions.

        **Reasons:**
        * **Mutual assistance:** improve survival chances among family with small size.
        * **Low efficiency:** may struggle to ensure everyone's safety among family with large size.
        """ )

elif field == 'port':
    with st.container():
        st.info("""
        **Observations:**
        * Port C with the highest proportion of class 1 also has the highest survival rate, but when it comes to port Q and port S, the distribution of pclass can not explain why port Q with almost all passengers from class 3 has a slight higer survival rate than port S. Therefore, there might be some other factors result in the contradiction.
        """)

elif field == 'age_group':
    with st.container():
        st.info("""
        **Explanations:**
        * **Children**: [0, 10]
        * **Teenager**: (10,18]
        * **Adult**: (18,60]      
        * **Senior**: (60,)
                
        **Observations:**
        * The bar chart shows a decline in survival rate as age increases and the survival rate of children(almost 60%) is nearly double of the senior group. What's more, as the adult passengers account for the majority of the samples, its survival rate(38.88%) is close to the overall(38.38%). 

        **Reasons:**
        * The reason might be the woman and children first principle.
        """ )

elif field == 'log_fare3':
    with st.container():
        st.info("""
        **Explanations:**
        * Group tickets were split to reflect individual fares, assuming children paid half the adult price.
        * Fare values were log-transformed to make the data distribution more suitable for analysis.
                    
        **Observations:**
        * The median fare of survivors is significantly higher than the died group. In the death group, there is a significantly wider distribution in the low fare price, and then drop rapidly when the price rise up.  
        """ )
st.markdown("---") 

# Exploration of Multiple Factors
st.subheader("Multiple Factor Exploration")
st.markdown("")
st.markdown("#### Exploration Ⅰ：Passenger Class and Gender Effects")
with st.expander("Observation Ⅰ: Fare Doesn't Matter in a Fixed Class", expanded=False):
    col_left, col_right = st.columns(2)
    with col_left:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        sns.violinplot(data=df,x='pclass',y='log_fare3', palette="viridis", ax=ax1)

        ax1.set_title("Fare Distribution by Pclass", pad=20)
        ax1.set_xlabel("pclass")
        ax1.set_ylabel("fare")
        st.pyplot(fig1)

    with col_right:
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
        sns.violinplot(x='survived',y='log_fare3',data=df[df.pclass == 1], palette="viridis", ax=ax1)

        ax1.set_title("Survival Effects of Fare in Pclass 1", pad=20)
        ax1.set_xlabel("Survived Status(0=No, 1=Yes)")
        ax1.set_ylabel("fare")
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.violinplot(x='survived',y='log_fare3',data=df[df.pclass == 2], palette="viridis", ax=ax2)

        ax2.set_title("Survival Effects of Fare in Pclass 2", pad=20)
        ax2.set_xlabel("Survived Status(0=No, 1=Yes)")
        ax2.set_ylabel("fare")
        st.pyplot(fig2)
    with col3:
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        sns.violinplot(x='survived',y='log_fare3',data=df[df.pclass == 3], palette="viridis", ax=ax3)

        ax3.set_title("Survival Effects of Fare in Pclass 3", pad=20)
        ax3.set_xlabel("Survived Status(0=No, 1=Yes)")
        ax3.set_ylabel("fare")
        st.pyplot(fig3)

with st.expander("Observation Ⅱ: Gender is the Dominant Survival Factor", expanded=False):
    col_left, col_right = st.columns(2)
    with col_left:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        sns.countplot(data=df,x='sex', hue='pclass', color='#66c2a5', ax=ax1)
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%d', padding=8)

        ax1.set_title("Sex Distribution by Pclass", pad=20)
        ax1.set_xlabel("sex")
        ax1.set_ylabel("count")
        st.pyplot(fig1)
    
    with col_right:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=df,x='sex',y='survived',hue='pclass', color='#66c2a5', errorbar=None, ax=ax2)
        for container in ax2.containers:
            ax2.bar_label(container, fmt=lambda x: f"{x*100:.1f}%", padding=8)

        ax2.set_title("Survival Rate by Gender and Pclass", pad=20)
        ax2.set_xlabel("sex")
        ax2.set_ylabel("survival rate (%)")
        st.pyplot(fig2)
    
    st.info("""
    **Observations:**
    * The survival rate of female in both class 1 and 2 is larger than 90% and even in class 3, females had a significantly higher chance to survive than males, and it seems that gender is a stronger impact than class. 
    """)
st.markdown("")

st.markdown("#### Exploration Ⅱ：Why Port Q (3rd Class Only) Has a Higher Survival Rate Than Port S?")
with st.expander("Observation: Port & Pclass", expanded=False):
    col_left, col_right = st.columns(2)
    with col_left:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        sns.countplot(data=df, x='port',hue='pclass', color='#66c2a5', ax=ax1)
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%d', padding=8)

        ax1.set_title("Port Distribution by Pclass", pad=20)
        ax1.set_xlabel("port")
        ax1.set_ylabel("count")
        st.pyplot(fig1)

    with col_right:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=df, x='port', y='survived', hue='pclass', color='#66c2a5', errorbar=None, ax=ax2)
        for container in ax2.containers:
            ax2.bar_label(container, fmt=lambda x: f"{x*100:.1f}%", padding=8)

        ax2.set_title("Survival Rate by Port and Pclass", pad=20)
        ax2.set_xlabel("port")
        ax2.set_ylabel("survival rate (%)")
        ax2.set_ylim(0, 1.1)
        st.pyplot(fig2)

    st.info("""
    **Phenomenon:**
    * Why do third-class passengers from Port Q have a higher survival rate than third-class passengers from Port S?
    """)

with st.expander('Assumption Ⅰ: Port & Sex'):
    col_left, col_right = st.columns(2)
    with col_left:
        df['sex_numeric'] = df.sex.apply(lambda x:1 if x == 'female' else 0)
        df_port_gender = df.groupby('port')['sex_numeric'].mean().sort_values(ascending=False).reset_index()
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=df_port_gender, x='port', y='sex_numeric', palette="viridis", ax=ax1)
        for container in ax1.containers:
            ax1.bar_label(container, fmt=lambda x: f"{x*100:.1f}%", padding=8)

        ax1.set_title('Female Proportion by Port', pad=20)
        ax1.set_xlabel('port')
        ax1.set_ylabel('female proportion (%)')
        ax1.set_ylim(0, 1.1)  
        st.pyplot(fig1)

    with col_right:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=df, x='sex', y='survived', hue='port', palette="viridis", errorbar=None, ax=ax2)
        for container in ax2.containers:
            ax2.bar_label(container, fmt=lambda x: f"{x*100:.1f}%", padding=8)

        ax2.set_title("Survival Rate by Gender and Port", pad=20)
        ax2.set_xlabel("sex")
        ax2.set_ylabel("survival rate (%)")
        ax2.set_ylim(0, 1.1)
        st.pyplot(fig2)

    st.info("""
    **Observations:**
    * The proportion of females in port Q is much higher than port S, and it may explain why survival rate in port Q(majority from class 3) is higher than port S as the gender has greater impact. 
    * In previous analysis of passenger class and gender, female passengers in class 3 show a significantly lower survival rate than other classes However, here females from port Q (almost all in class 3) have a higher survival rate than females from Port S.
    """)

with st.expander('Assumption Ⅱ: Port & Sex & Age'):
    col_left, col_right = st.columns(2)
    with col_left:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        sns.violinplot(data=df,x='port',y='age_filled',palette="viridis", ax=ax1)

        ax1.set_title('Age Distribution by Port', pad=20)
        ax1.set_xlabel('port')
        ax1.set_ylabel('age')
        st.pyplot(fig1)

    with col_right:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        sns.violinplot(data=df[df.sex == 'female'],x='port',y='age_filled', palette="viridis", ax=ax2)
        
        ax2.set_title("Age Distribution of Female Passengers by Port", pad=20)
        ax2.set_xlabel("port")
        ax2.set_ylabel("female age")
        st.pyplot(fig2)

    st.info("""
    **Observations:**
    * Female passengers in port Q is significantly lower than other ports, it might be the reason why female in port Q has higher survival rate. However, the sample of female in port Q is small(36). 
            
    **Next step:**
    * Find the relationship between female survival rate within different age to examine whether younger women has higher survival rate.
    """)

    col_left, col_right = st.columns(2)
    with col_left:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        sns.histplot(data=df.age[df.sex == 'female'], palette="viridis", ax=ax1)

        ax1.set_title('Distribution of Age of Female', pad=20)
        ax1.set_xlabel('age')
        ax1.set_ylabel('count')
        st.pyplot(fig1)

    with col_right:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        sns.violinplot(data=df[df.sex == 'female'],x='survived',y='age_filled', palette="viridis", ax=ax2)
        
        ax2.set_title("Distribution of Female Survival Rate by Age", pad=20)
        ax2.set_xlabel("Survived Status(0=No, 1=Yes)")
        ax2.set_ylabel("female age")
        st.pyplot(fig2)
    
    st.info("""
    **Observations:**
    * The age advantage seems to be a specific factor in port Q, the overall distribution shows that most of the survived females are older than those died. The remaining variable is the family size.
    """)

with st.expander('Assumption Ⅲ: Port & Sex & Age & Family Size'):
    col_left, col_right = st.columns(2)
    with col_left:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        sns.countplot(data=df[df.sex == 'female'],x='port',hue='family_size', palette="viridis", ax=ax1)
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%d', padding=8)

        ax1.set_title('The Distribution of Females in Different Family Size by Port', pad=20)
        ax1.set_xlabel('port')
        ax1.set_ylabel('count')
        ax1.legend(title="Family Size", loc='upper right', fontsize=8)
        st.pyplot(fig1)

    with col_right:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=df[df.sex == 'female'],x='port',y='survived',hue='family_size',errorbar=None, palette="viridis", ax=ax2)
        for container in ax2.containers:
            ax2.bar_label(container, fmt=lambda x: f"{x*100:.1f}%", padding=8)

        ax2.set_title("The Survival Rate of Females in Different Family Size by Port", pad=20)
        ax2.set_xlabel("port")
        ax2.set_ylabel("survival rate (%)")
        ax1.set_ylim(0, 1.1)
        ax2.legend(title="Family Size", loc='lower right', fontsize=8)
        st.pyplot(fig2)

    st.info("""
    **Observations:**
    * The reason for the higher survival rate for female in port Q seems related with the family size, it shows that almost no female from large family size in port Q while port S have the highest number of females from large family. 
    """)
st.markdown("---") 

# Conclusions
st.subheader("Conclusion")
st.info("""
    * In conclusion, **Gender** was the **most important factor** for survival, followed by **Passenger Class**. **Age** protected the children, and **Family Size** provided a survival advantage for small groups. Finally **Fare** was highly correlated with class but had almost no impact on survival within the same passenger class.
    """)

















