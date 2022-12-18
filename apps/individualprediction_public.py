# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 19:07:35 2022

@author: Hyeonji Oh
"""

# import libraries needed
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import lifelines
import sksurv.ensemble
import cloudpickle as cp
from urllib.request import urlopen


#Import url links from public google drive
aft_url = 'https://drive.google.com/file/d/1QB2F1onzrnQUfgO7HUulxFlRmWHvr7zH/view?usp=sharing'
aft_url = 'https://drive.google.com/uc?export=download&id=' + aft_url.split('/')[-2]

cph_url = 'https://drive.google.com/file/d/1izPfWwpTCUa8sHCzK0S-jl1tCRNeDTvM/view?usp=sharing'
cph_url = 'https://drive.google.com/uc?id=' + cph_url.split('/')[-2]

rsf_url = 'https://drive.google.com/file/d/1roCsCPRHIMhwiiE46WR5Zdc8kE8_MW3S/view?usp=share_link'
rsf_url = 'https://drive.google.com/uc?id=' + rsf_url.split('/')[-2]

aft_median_url = 'https://drive.google.com/file/d/1CgNcglj7IiURDBa2AnBa_ge1j-aRvbUg/view?usp=share_link'
aft_median_url = 'https://drive.google.com/uc?id=' + aft_median_url.split('/')[-2]

cph_median_url = 'https://drive.google.com/file/d/1R2WvjvfELq8G9C41QD6SHKhY7_x-eDQ5/view?usp=share_link'
cph_median_url = 'https://drive.google.com/uc?id=' + cph_median_url.split('/')[-2]

rsf_median_url = 'https://drive.google.com/file/d/1--q4Xh5e_95RR2wZQKF0EZ2tzIFyfMkf/view?usp=share_link'
rsf_median_url = 'https://drive.google.com/uc?id=' + rsf_median_url.split('/')[-2]

df_train_url = 'https://drive.google.com/file/d/1T-Pw0Znu9mbbqwjS2rCOj_gegm27MvjO/view?usp=share_link'
df_train_url = 'https://drive.google.com/uc?id=' + df_train_url.split('/')[-2]


# loading the saved model
aft = cp.load(urlopen(aft_url))
cph = cp.load(urlopen(cph_url))
rsf = cp.load(urlopen(rsf_url))

df_train = pd.read_csv(df_train_url)
cph_median = pd.read_csv(cph_median_url)
aft_median = pd.read_csv(aft_median_url)
rsf_median = pd.read_csv(rsf_median_url)
    
aft_final_feature_list = ['mean_bulbar', 'slope_bulbar', 'Age', 'mean_fvc', 'onset_delta', 'slope_weight', 'slope_ALSFRS_R_Total', 'mean_Q5_Cutting', 'mean_Creatinine']
cph_final_feature_list = ['mean_bulbar', 'slope_bulbar', 'slope_ALSFRS_R_Total', 'Age', 'slope_weight', 'onset_delta', 'mean_fvc', 'mean_ALSFRS_R_Total']
rsf_final_feature_list = ['mean_bulbar', 'slope_ALSFRS_R_Total', 'mean_ALSFRS_R_Total', 'onset_delta', 'mean_fvc', 'slope_weight','Age', 'mean_weight'] 
    

def predict_rsf_percentile(data, percentile):
    result_per = rsf.predict_survival_function(data.to_numpy().reshape(1, -1), return_array = True)
    result_per = np.squeeze(result_per)
    time_result = pd.DataFrame({'time' : rsf.event_times_, 'p' : result_per })
    if time_result[time_result['p'] <= percentile].count()['time'] == 0:
      per = np.inf
    else:
      per = time_result[time_result['p'] <= percentile].iloc[0,0]
    
    return per


def app():

    
    # Preprocessing average curve
    aft_slow_list = list(aft_median[aft_median['0'] >= 41.537768]['SubjectID'])
    aft_intermediate_list = list(aft_median[(aft_median['0'] >= 18.924361) & (aft_median['0'] < 41.537768)]['SubjectID'])
    aft_rapid_list = list(aft_median[aft_median['0']  < 18.924361]['SubjectID'])
    X_aft_slow = df_train[df_train['SubjectID'].isin(aft_slow_list)][aft_final_feature_list]
    X_aft_intermediate = df_train[df_train['SubjectID'].isin(aft_intermediate_list)][aft_final_feature_list]
    X_aft_rapid = df_train[df_train['SubjectID'].isin(aft_rapid_list)][aft_final_feature_list]
    X_aft_full = df_train[aft_final_feature_list]
    
    result_aft_slow = pd.DataFrame(aft.predict_survival_function(X_aft_slow.iloc[:,:]).mean(axis=1))
    result_aft_intermediate = pd.DataFrame(aft.predict_survival_function(X_aft_intermediate.iloc[:,:]).mean(axis=1))
    result_aft_rapid = pd.DataFrame(aft.predict_survival_function(X_aft_rapid.iloc[:,:]).mean(axis=1))
    result_aft_full = pd.DataFrame(aft.predict_survival_function(X_aft_full.iloc[:,:]).mean(axis=1))

    cph_slow_list = list(cph_median[cph_median['0.5'] >= 	41.690000]['SubjectID'])
    cph_intermediate_list = list(cph_median[(cph_median['0.5'] >= 17.230000) & (cph_median['0.5'] < 41.690000)]['SubjectID'])
    cph_rapid_list = list(cph_median[cph_median['0.5']  <  17.230000]['SubjectID'])
    X_cph_slow = df_train[df_train['SubjectID'].isin(cph_slow_list)][cph_final_feature_list]
    X_cph_intermediate = df_train[df_train['SubjectID'].isin(cph_intermediate_list)][cph_final_feature_list]
    X_cph_rapid = df_train[df_train['SubjectID'].isin(cph_rapid_list)][cph_final_feature_list]
    X_cph_full = df_train[cph_final_feature_list]
    
    result_cph_slow = pd.DataFrame(cph.predict_survival_function(X_cph_slow.iloc[:,:]).mean(axis=1))
    result_cph_intermediate = pd.DataFrame(cph.predict_survival_function(X_cph_intermediate.iloc[:,:]).mean(axis=1))
    result_cph_rapid = pd.DataFrame(cph.predict_survival_function(X_cph_rapid.iloc[:,:]).mean(axis=1))
    result_cph_full = pd.DataFrame(cph.predict_survival_function(X_cph_full.iloc[:,:]).mean(axis=1))
        
    rsf_slow_list = list(rsf_median[rsf_median['0']>=74.610988]['SubjectID'])
    rsf_intermediate_list = list(rsf_median[rsf_median['0']>=16.73]['SubjectID'])
    rsf_rapid_list = list(rsf_median[rsf_median['0']<16.73]['SubjectID'])
    
    X_rsf_slow = df_train[df_train['SubjectID'].isin(rsf_slow_list)]
    X_rsf_intermediate = df_train[df_train['SubjectID'].isin(rsf_intermediate_list)]
    X_rsf_rapid = df_train[df_train['SubjectID'].isin(rsf_rapid_list)]
    X_rsf_full = df_train[rsf_final_feature_list]
    
    rsf_pred_slow = np.squeeze(rsf.predict_survival_function(X_rsf_slow[rsf_final_feature_list].iloc[:, :].to_numpy(), return_array = True))
    rsf_pred_intermediate = np.squeeze(rsf.predict_survival_function(X_rsf_intermediate[rsf_final_feature_list].iloc[:, :].to_numpy(), return_array = True))
    rsf_pred_rapid = np.squeeze(rsf.predict_survival_function(X_rsf_rapid[rsf_final_feature_list].iloc[:, :].to_numpy(), return_array = True))
    rsf_pred_full = np.squeeze(rsf.predict_survival_function(X_rsf_full[rsf_final_feature_list].iloc[:,:].to_numpy(),return_array = True))
    
    result_rsf_slow = np.transpose(pd.DataFrame(rsf_pred_slow)).set_index(rsf.event_times_)
    result_rsf_intermediate = np.transpose(pd.DataFrame(rsf_pred_intermediate)).set_index(rsf.event_times_)
    result_rsf_rapid = np.transpose(pd.DataFrame(rsf_pred_rapid)).set_index(rsf.event_times_)
    result_rsf_full = np.transpose(pd.DataFrame(rsf_pred_full)).set_index(rsf.event_times_)
    
    result_rsf_slow = pd.DataFrame(result_rsf_slow.mean(axis=1))
    result_rsf_intermediate = pd.DataFrame(result_rsf_intermediate.mean(axis=1))
    result_rsf_rapid =  pd.DataFrame(result_rsf_rapid.mean(axis=1))
    result_rsf_full = pd.DataFrame(result_rsf_full.mean(axis=1))
    

    
    # loading navigation bar style
    st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
    # create blue navigation bar
    st.markdown("""
    <nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #153888;">
      <a class="navbar-brand" href="#" target="_blank">LASF prediction model</a>
    """, unsafe_allow_html=True)
    
    header = st.container()
    model_training = st.container()
    container = st.container()
    
    with model_training:
        # Create space for feature input in sidebar
        st.sidebar.title('Features')
        st.sidebar.write('If any values are unknown in user data, please input the existing default value.')
        st.markdown( """ <style> .sidebar .sidebar-content { background-image: linear-gradient(#DAE3F3,#DAE3F3); color: blue; } </style> """, unsafe_allow_html=True, )
        Age = st.sidebar.text_input('Age (years)', value=56)
        mean_ALSFRS_R_Total = st.sidebar.slider('mean ALSFRS-R total score (points)', min_value=20, max_value = 50, value=40, step=5)
        mean_bulbar = st.sidebar.slider('mean ALSFRS-R bulbar score (points)', min_value=5, max_value = 10, value=8, step=1)
        mean_Q5_Cutting = st.sidebar.slider('mean ALSFRS-R Q5_Cutting (points)', min_value=1, max_value = 4, value=2, step=1)
        mean_fvc = st.sidebar.slider('mean FVC (%)', min_value=10, max_value = 100, value=70, step=10)
        mean_Creatinine = st.sidebar.slider('mean Creatinine (mmol/L)', min_value=10, max_value = 100, value=50, step=10)
        onset_delta = st.sidebar.text_input('Time from onset (months)', value=-20)
        slope_ALSFRS_R_Total = st.sidebar.text_input('ALSFRS-R total slope (points per month)', value=-1)
        slope_bulbar = st.sidebar.text_input('ALSFRS-R bulbar slope (points per month)', value=-1)
        mean_weight = st.sidebar.text_input('mean weight (kg)', value=60)
        slope_weight = st.sidebar.text_input('weight change rate (kg per month)', value=-1)
        
        Age_calculated = (float(Age)-float(15))//5
        
        # patient info filled with input value
        patient_aft = pd.DataFrame({'mean_bulbar':mean_bulbar, 'slope_bulbar':slope_bulbar, 'Age':Age_calculated, 'mean_fvc':mean_fvc, 'onset_delta':onset_delta, 'slope_weight':slope_weight, 'slope_ALSFRS_R_Total':slope_ALSFRS_R_Total, 'mean_Q5_Cutting':mean_Q5_Cutting, 'mean_Creatinine':mean_Creatinine}, index = ['Current patient'])
        patient_cph = pd.DataFrame({'mean_bulbar':mean_bulbar, 'slope_bulbar':slope_bulbar, 'slope_ALSFRS_R_Total':slope_ALSFRS_R_Total, 'Age':Age_calculated, 'slope_weight':slope_weight, 'onset_delta':onset_delta, 'mean_fvc':mean_fvc, 'mean_ALSFRS_R_Total':mean_ALSFRS_R_Total}, index = ['Current patient'])
        patient_rsf = pd.DataFrame({'mean_bulbar':mean_bulbar, 'slope_ALSFRS_R_Total':slope_ALSFRS_R_Total, 'mean_ALSFRS_R_Total':mean_ALSFRS_R_Total, 'onset_delta':onset_delta, 'mean_fvc':mean_fvc, 'slope_weight':slope_weight,'Age':Age_calculated, 'mean_weight':mean_weight}, index = ['Current patient'])
        show_average = []
        show_progressor = []
        
    with container:
        
        option = st.selectbox('Select model type',('Accelerated failure time', 
                       'Cox proportional hazard', 'Random survival forests'))

        
        model_name_dic = {"Accelerated failure time":[aft, patient_aft, result_aft_full], "Cox proportional hazard":[cph, patient_cph, result_cph_full], "Random survival forests":[rsf, patient_rsf, result_rsf_full]}
        progressor_dic = {"Accelerated failure time":[result_aft_slow, result_aft_intermediate, result_aft_rapid], "Cox proportional hazard":[result_cph_slow, result_cph_intermediate,result_cph_rapid], "Random survival forests":[result_rsf_slow, result_rsf_intermediate, result_rsf_rapid]}
        
        selected_model = model_name_dic[option][0]
        selected_patient = model_name_dic[option][1]
        selected_average = model_name_dic[option][2]
        
        result_slow = progressor_dic[option][0]
        result_intermediate = progressor_dic[option][1]
        result_rapid = progressor_dic[option][2]
        
        
        check_col1, check_col2, empty_col = st.columns(3)
        with check_col1:
            check_average = st.checkbox('Show average curve')
        
        if check_average:
            show_average.append(1)
            
        with check_col2:
            check_group = st.checkbox('Show progressor group')
        if check_group:
            show_progressor.append(1)
            
            
    fig, ax = plt.subplots(figsize=(25,14))       
    if show_average==[1] and show_progressor==[1]: 
        ax = plt.plot(result_slow.index, result_slow[0], marker='None', color='#C0C0C0', linestyle="--", linewidth=1.8, label='Slow')
        ax = plt.plot(result_intermediate.index, result_intermediate[0], marker='None', color='#696969', linestyle="--", linewidth=1.8, label='Intermediate')
        ax = plt.plot(result_rapid.index, result_rapid[0], marker='None', color='k', linestyle="--", linewidth=1.8, label='Rapid')
        ax = plt.plot(selected_average.index, selected_average[0], marker='None', color='blue', linestyle="--", linewidth=1.8, label='Average')
        if selected_model == rsf:
            result_rsf = rsf.predict_survival_function(selected_patient, return_array=True)
            for i, s in enumerate(result_rsf):
                plt.step(rsf.event_times_, s, where="post")
                plt.legend(labels = list(patient_rsf.index))
                plt.xlabel("Time in months")
                plt.ylim([0,1.08])
            st.pyplot(fig)
        else:
            result= selected_model.predict_survival_function(selected_patient)
            ax = sns.lineplot(data = result)
            ax.set(xlabel='Time in months', ylabel='S(t)')
            st.pyplot(fig) 
                  
    elif show_average==[1] and show_progressor==[]:
        ax = plt.plot(selected_average.index, selected_average[0], marker='None', color='blue', linestyle="--", linewidth=1.8, label='Average')
        if selected_model == rsf:
            result_rsf = rsf.predict_survival_function(selected_patient, return_array=True)
            for i, s in enumerate(result_rsf):
                plt.step(rsf.event_times_, s, where="post")
                plt.legend(labels = list(patient_rsf.index))
                plt.xlabel("Time in months")
                plt.ylim([0,1.08])
            st.pyplot(fig)
        else:
            result= selected_model.predict_survival_function(selected_patient)
            ax = sns.lineplot(data = result)
            ax.set(xlabel='Time in months', ylabel='S(t)')
            st.pyplot(fig)
            
    elif show_average==[] and show_progressor==[1]:
        ax = plt.plot(result_slow.index, result_slow[0], marker='None', color='#C0C0C0', linestyle="--", linewidth=1.8, label='Slow')
        ax = plt.plot(result_intermediate.index, result_intermediate[0], marker='None', color='#696969', linestyle="--", linewidth=1.8, label='Intermediate')
        ax = plt.plot(result_rapid.index, result_rapid[0], marker='None', color='k', linestyle="--", linewidth=1.8, label='Rapid')
        if selected_model == rsf:
            result_rsf = rsf.predict_survival_function(selected_patient, return_array=True)
            for i, s in enumerate(result_rsf):
                plt.step(rsf.event_times_, s, where="post")
                plt.legend(labels = list(patient_rsf.index))
                plt.xlabel("Time in months")
                plt.ylim([0,1.08])
            st.pyplot(fig)
        else:
            result= selected_model.predict_survival_function(selected_patient)
            ax = sns.lineplot(data = result)
            ax.set(xlabel='Time in months', ylabel='S(t)')
            st.pyplot(fig) 
    else:
        if selected_model == rsf:
            result_rsf = rsf.predict_survival_function(selected_patient, return_array=True)
            for i, s in enumerate(result_rsf):
                plt.step(rsf.event_times_, s, where="post")
                plt.legend(labels = list(patient_rsf.index))
                plt.xlabel("Time in months")
                plt.ylim([0,1.08])
            st.pyplot(fig)
        else:
            result= selected_model.predict_survival_function(selected_patient)
            ax = sns.lineplot(data = result)
            ax.set(xlabel='Time in months', ylabel='S(t)')
            st.pyplot(fig)         

    probability = st.slider('Predict probability point (%)', min_value=30, max_value = 100, value=60, step=1)
    if selected_model == rsf:
        time2 = predict_rsf_percentile(patient_rsf,0.01*probability)
        if time2 == np.inf:
            function = pd.DataFrame(rsf.predict_survival_function(patient_rsf[rsf_final_feature_list].iloc[:, :].to_numpy(), return_array = True))
            function.columns = rsf.event_times_
            last_point = pd.DataFrame(function.iloc[:, -1])
            linear = (0.5*last_point.columns[0])/(1-last_point.iloc[:,0])
            time = linear.values[0]
        else:
            time = time2
        st.write(str(round(float(time),2))+' months from past 3 months (lineary extended)')
    else:
        time = selected_model.predict_percentile(selected_patient, p=0.01*probability)
        st.write(str(round(float(time),2))+' months from past 3 months')
        


