# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 19:07:35 2022

@author: Hyeonji Oh
"""

# import libraries needed
import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from apps import home, individualprediction, contact_us, how_to_use
import database as db


# loading navigation bar stylesheet
st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

# create navigation bar
st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #153888;">
  <a class="navbar-brand" href="#" target="_blank">LASF prediction model</a>
</nav>
""", unsafe_allow_html=True)

#--- USER AUTHENTICATION ---
users = db.fetch_all_users() 
    
usernames = [user["key"] for user in users]
names = [user["name"] for user in users]
hashed_passwords = [user["password"] for user in users]


credentials = {"usernames":{}}

for un, name, pw in zip(usernames, names, hashed_passwords):
    user_dict = {"name":name,"password":pw}
    credentials["usernames"].update({un:user_dict})
    
authenticator = stauth.Authenticate(credentials, "sales_dashboard", "abcdef", cookie_expiry_days=0)

name, authentication_status, username = authenticator.login("Login", "sidebar")



if authentication_status == False:
    st.error("Username/password is incorrect or your signup is pending approval")
    
    
if authentication_status == None:
    st.warning("Please enter your username and password to access individual prediction")
    
    st.sidebar.write("If you don't have account, please signup")
    
    username_signup = st.sidebar.text_input(label="User Name", value="")
    name_signup = st.sidebar.text_input(label="Name", value="")
    password_signup = st.sidebar.text_input(label="Password", value="")
    checkbox = st.sidebar.checkbox('Are you clinician?')
    btn_clicked = st.sidebar.button("Signup", disabled=(checkbox is False))
    con = st.sidebar.container()
    
    if btn_clicked:
        if not str(username_signup):
            con.error("Input your name please~")
        
        elif not str(name_signup):
            con.error("Input your name please~")
            
        elif not str(password_signup):
            con.error("Input your password please~")
        
        else:
            hashed_passwords_signup = stauth.Hasher(password_signup).generate()[0]

            db.insert_user(username_signup, name_signup, hashed_passwords_signup)
            
            
            
        
    PAGES = {"Home":home, "How to use":how_to_use, "Contact us":contact_us}
    selection = st.radio('Go to', list(PAGES.keys()))
    page = PAGES[selection]
    page.app()
    

if authentication_status:
    auth_status = 1
    
    PAGES = {"Home":home, "How to use":how_to_use, "Contact us":contact_us, "Individual prediction": individualprediction}
    selection = st.radio('Go to', list(PAGES.keys()))
    page = PAGES[selection]
    page.app()

    authenticator.logout("Logout", "sidebar")
    st.sidebar.title(f"Welcome {name}")
