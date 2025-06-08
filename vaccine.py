import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sklearn

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore',category=UserWarning,module='matplotlib')


st.set_page_config(layout='wide')

st.title("H1N1 Vaccine Usage Prediction")

# reading the required files

df = pd.read_csv(r'S:/Projects/vaccine_usage_prediction/vac_final.csv')

with open (r'S:/Projects/vaccine_usage_prediction/model_log_reg.pkl','rb') as f:
    model = pickle.load(f)


tab1,tab2,tab3 = st.tabs(['About','Analysis','Model Prediction'])

with tab1:
    st.header('**H1N1 Virus:**')
    st.write("The H1N1 virus, also known as swine flu, is a subtype of influenza A virus that can infect humans, pigs, and birds. It caused a global pandemic in 2009, leading to an estimated 284,400 deaths worldwide. The virus spreads through airborne droplets when an infected person coughs, sneezes, or talks.")
    st.header("**Key Facts:**")
    st.write("- Symptoms: Fever, cough, sore throat, body aches, chills, fatigue, and sometimes vomiting or diarrhea.")
    st.write("- Transmission: Person-to-person via respiratory droplets or contaminated surfaces.")
    st.write("- Prevention: Seasonal flu vaccines now include protection against H1N1.")
    st.write("- Treatment: Rest, fluids, and antiviral medications like oseltamivir (Tamiflu) and zanamivir (Relenza)") 

    st.header('ML Model:')
    st.write('I have used Logistic Regression Model for this Project. The Model uses 33 user inputs to predict the output.')

    st.header('User Input:')
    st.write('- h1n1_worry - Worry about the h1n1 flu(0,1,2,3) 0=Not worried at all, 1=Not very worried, 2=Somewhat worried, 3=Very worried')
    st.write('h1n1_awareness - Signifies the amount of knowledge or understanding the respondent has about h1n1 flu - (0,1,2) - 0=No knowledge, 1=little knowledge, 2=good knowledge')
    
  
with tab2:

    st.header('Correlation Heatmap:')

    corr_df = df.corr(method='spearman')
    df_corr = plt.figure(figsize=(20,15))
    sns.heatmap(corr_df,annot=True,cmap='coolwarm',fmt='0.1f')
    plt.title('Spearman Rank Correlation Heatmap')
    st.pyplot(df_corr)

    col1,col2 = st.columns(2)

    with col1:
        st.header('Population which took the vaccine:')
        st.write('0 -> Vaccine not taken')
        st.write('1 -> Vaccine taken')
        fig,ax = plt.subplots(figsize =(8,6))
        plot_vacc_pop = df['h1n1_vaccine'].value_counts(normalize=True).plot(kind='bar',ax=ax)
        ax.set_label("Vaccination Status (0 = Not Taken, 1 = Taken)")
        st.pyplot(fig)

        st.header('Population which has chronic disease:')

        st.write("0 -> People with chronic disease didn't take the vaccine")
        st.write("1 -> People with chronic disease took vaccine")
        fig_chr,ax = plt.subplots(figsize = (8,6))
        plot_chr_di = sns.countplot(x='chronic_medic_condition',hue='h1n1_vaccine',data=df)
        st.pyplot(fig_chr)

        st.header('Vaccination based on Age Factor:')

        st.write('0 -> 18 - 34 years')
        st.write('1 -> 35 - 44 years')
        st.write('2 -> 45 - 54 years')
        st.write('3 -> 55 - 64 years')
        st.write('4 - 65+ years')

        fig_age, ax = plt.subplots(figsize = (8,6))
        plot_age = sns.countplot(x='age_bracket',hue='h1n1_vaccine',data=df)
        st.pyplot(fig_age)

    with col2:

        st.header('Vaccination based on Gender:')

        st.write("0 -> Female")
        st.write("1 -> Male")
        fig_gen, ax = plt.subplots(figsize = (8,6))
        plot_gen = sns.countplot(x='sex',hue='h1n1_vaccine',data=df)
        st.pyplot(fig_gen)

        st.header('H1N1 awarness among the population:')

        st.write('0 -> no awarness')
        st.write('1 -> little awarness')
        st.write('2 -> enough awarness')
        fig_aw, ax = plt.subplots(figsize = (8,6))
        plot_aw = sns.countplot(x='h1n1_awareness',hue='h1n1_vaccine',data=df)
        st.pyplot(fig_aw)

with tab3:
    st.title('**Vaccine Usage Prediction**')
    col1,col2 = st.columns(2)
    with col1:
        h1n1_worry = st.radio("H1N1 Worry Status : (0=Not worried at all, 1=Not very worried, 2=Somewhat worried, 3=Very worried)",(0,1,2,3))
        h1n1_awareness = st.radio("H1N1 Awareness Status :(0=No knowledge, 1=little knowledge, 2=good knowledge)",(0,1,2))
        antiviral_medication = st.radio("Have You Previously taken any antiviral medication : (0=No,1=Yes)",(0,1))
        contact_avoidance = st.radio("Have you avoided contact with people who has flu : (0=No,1=Yes)",(0,1))
        bought_face_mask = st.radio("Have you bought a face mask : (0=No,1=Yes)",(0,1))
        wash_hands_frequently = st.radio("Do you wash hands frequently : (0=No,1=Yes)",(0,1))
        avoid_large_gatherings = st.radio("Do you avoid large gatherings : (0=No,1=Yes)",(0,1))
        reduced_outside_home_cont = st.radio("Have you reduced contact with people outside home : (0=No,1=Yes)",(0,1))
        avoid_touch_face = st.radio("Did you avoid touching nose, eyes and mouth : (0=No,1=Yes)",(0,1))
        dr_recc_h1n1_vacc = st.radio("Has the Doctor recommended H1N1 vaccine : (0=No,1=Yes)",(0,1))
        dr_recc_seasonal_vacc = st.radio("Has the Doctor recommended seasonal flu vaccine : (0=No,1=Yes)",(0,1))
        chronic_medic_condition = st.radio("Do you have any chronic medical condition : (0=No,1=Yes)",(0,1))
        cont_child_undr_6_mnths = st.radio("Are you in contact with child under 6 months : (0=No,1=Yes)",(0,1))
        is_health_worker = st.radio("Are you a health worker : (0=No,1=Yes)",(0,1))
        has_health_insur = st.radio("Do you have a health insurance : (0=No,1=Yes)",(0,1))
        is_h1n1_vacc_effective = st.radio("Do you think if the H1N1 Vaccine is effective : (1=Thinks not effective at all, 2=Thinks it is not very effective, 3=Doesn't know if it is effective or not, 4=Thinks it is somewhat effective, 5=Thinks it is highly effective)",(1,2,3,4,5))
        is_h1n1_risky = st.radio("Do you think the H1N1 virus is risky in the absence of vaccine : (1=Thinks it is not very low risk, 2=Thinks it is somewhat low risk, 3=don’t know if it is risky or not, 4=Thinks it is a somewhat high risk, 5=Thinks it is very highly risky)",(1,2,3,4,5))
        sick_from_h1n1_vacc = st.radio("Do you worry if the H1N1 vaccine would make you sick : (1=Respondent not worried at all, 2=Respondent is not very worried, 3=Doesn't know, 4=Respondent is somewhat worried, 5=Respondent is very worried)",(1,2,3,4,5))

    with col2:
    
        is_seas_vacc_effective = st.radio("Do you think if the seasonal vaccine is effective : (1=Thinks not effective at all, 2=Thinks it is not very effective, 3=Doesn't know if it is effective or not, 4=Thinks it is somewhat effective, 5=Thinks it is highly effective)",(1,2,3,4,5))
        is_seas_risky = st.radio("Do you worry on getting ill due to seasonal flu in the absence of vaccine : (1=Thinks it is not very low risk, 2=Thinks it is somewhat low risk, 3=Doesn't know if it is risky or not, 4=Thinks it is somewhat high risk, 5=Thinks it is very highly risky)",(1,2,3,4,5))
        sick_from_seas_vacc = st.radio("Do you think you will get sick by taking the seasonal flu vaccine : (1=Respondent not worried at all, 2=Respondent is not very worried, 3=Doesn't know, 4=Respondent is somewhat worried, 5=Respondent is very worried)",(1,2,3,4,5))
        age_bracket = st.radio("What is your age : (0= 8 - 34 Years, 1= 35 - 44 Years, 2= 45 - 54 Years, 3= 55 - 64 Years, 4= 64+ Years)",(0,1,2,3,4))
        qualification = st.radio("What is your qualification : (0 = Some College,1 = College Graduate,2 = 12 Years,3 = < 12 Years)",(0,1,2,3))
        race = st.radio("What is your race : (0 = Black,1 = Other or Multiple,2 = Black,3 = White)",(0,1,2,3))
        sex = st.radio("What is your sex : (0 = Women, 1 = Men)",(0,1))
        income_level = st.radio(" Your income level : (0 = > $75,000 , 1 = <= $75,000, 2 = Below Poverty)",(0,1,2))
        marital_status = st.radio("Your Relationship status : (0 = Not Married, 1 = Married)",(0,1))
        housing_status = st. radio("Is your premisis : (0 = own, 1 = rent)",(0,1))
        employment = st.radio("Your employment status : (0 = Not in Labor Force, 1 = Employed, 2 = Unemployed)",(0,1,2))
        census_msa = st.radio("Your Residence : (0 = MSA, Principle City, 1 = Non-MSA, 2 = MSA, Not Principle  City)",(0,1,2))
        no_of_adults = st.number_input("Number of adults in your household : ")
        no_of_children = st.number_input("Number of children in your household : ")


    submit_button = st.button("Submit",use_container_width=True)
    if submit_button:
        input_data = pd.DataFrame({
            'h1n1_worry' : [h1n1_worry],
            'h1n1_awareness' : [h1n1_awareness],
            'antiviral_medication' : [antiviral_medication],
            'contact_avoidance' : [contact_avoidance],
            'bought_face_mask' : [bought_face_mask],
            'wash_hands_frequently' : [wash_hands_frequently],
            'avoid_large_gatherings' : [avoid_large_gatherings],
            'reduced_outside_home_cont' : [reduced_outside_home_cont],
            'avoid_touch_face' : [avoid_touch_face],
            'dr_recc_h1n1_vacc' : [dr_recc_h1n1_vacc],
            'dr_recc_seasonal_vacc' : [dr_recc_seasonal_vacc],
            'chronic_medic_condition' : [chronic_medic_condition],
            'cont_child_undr_6_mnths' : [cont_child_undr_6_mnths],
            'is_health_worker' : [is_health_worker],
            'has_health_insur' : [has_health_insur],
            'is_h1n1_vacc_effective' : [is_h1n1_vacc_effective],
            'is_h1n1_risky' : [is_h1n1_risky],
            'sick_from_h1n1_vacc' : [sick_from_h1n1_vacc],
            'is_seas_vacc_effective' : [is_seas_vacc_effective],
            'is_seas_risky' : [is_seas_risky],
            'sick_from_seas_vacc' : [sick_from_seas_vacc],
            'age_bracket' : [age_bracket],
            'qualification' : [qualification],
            'race' : [race],
            'sex' : [sex],
            'income_level' : [income_level],
            'marital_status' : [marital_status],
            'housing_status' : [housing_status],
            'employment' : [employment],
            'census_msa' : [census_msa],
            'no_of_adults' : [no_of_adults],
            'no_of_children' : [no_of_children]})
        
        prediction = model.predict(input_data)

        st.subheader ("H1N1 Vaccine Recommendation :")

        if prediction[0] == 1:
            st.success("The Vaccine is Recommended")
        else:
            st.success("The Vaccine is not Recommended")

