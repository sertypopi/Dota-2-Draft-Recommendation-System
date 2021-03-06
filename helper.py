import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import os
import pandas as pd
from model_funcs_opp import predictions


import numpy as np

#session always gets refreshed when a button is clicked 

#st.session_state is a dictionary which persists even if a button is clicked
#currently picked is a key whose value is the currently selected heroes

if 'currently_picked' not in st.session_state:
    st.session_state['currently_picked']=[]
    

#FUNCTION TO FILL st.session_state when a hero is selected
def hero_selected(ind):
    #captains mode sequence
    picks_order=[0,1,1,0,1,0,0,1,0,1]
    if len(st.session_state['currently_picked'])<9:
        st.session_state['currently_picked'].append(ind)
        for i,j in enumerate(st.session_state['currently_picked']):
            if picks_order[i]==0:
                col6.image(".\\heroes\\"+hero_df[hero_df['sid']==j]['image_names'].values[0])
            else:
                col7.image(".\\heroes\\"+hero_df[hero_df['sid']==j]['image_names'].values[0])
    else:
        for i,j in enumerate(st.session_state['currently_picked']):
            if picks_order[i]==0:
                col6.image(".\\heroes\\"+hero_df[hero_df['sid']==j]['image_names'].values[0])
            else:
                col7.image(".\\heroes\\"+hero_df[hero_df['sid']==j]['image_names'].values[0])

#Funtion to edit markdown elements accordings to stlyes.css file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)








st.title("Dota 2 Draft Recommendation System")
for i in range(4):
    st.write('\n')
hero_df=pd.read_csv('hero_df.csv')
hero_list=os.listdir(".\\heroes\\")
hero=0
picks_order=[0,1,1,0,1,0,0,1,0,1]
#Defining columns and estabilishing size relation
col1,col2,col3,col4,col5,gap,col6,gap2,gap3,col7,col8,col9=st.columns([3,3,3,3,3,1,3,1,1,3,1,1])


#Filling 24 rows and 5 columns with hero images and buttons
for i in range(24):
    for j in [col1,col2,col3,col4,col5]:
        j.image('.\\heroes\\'+hero_df[hero_df['sid']==hero]['image_names'].values[0])
        #Creating button under each hero
        # on_click argument specifies the name of the function to execute when that button is clicked.args is the parameter
        # to pass to that function  
        j.button(hero_df.loc[hero_df['sid']==hero]['localized_name'].values[0],key=hero,on_click=hero_selected,args=[hero])
        hero+=1

local_css("style.css")


col6.subheader("Radiant")
col7.subheader("Dire")
for i in range(5):
    col8.write('\n')




clear=col8.button("Clear")
predict=col8.button("Predict",key='prediction')

#if clear is clicked execute this 
if clear:
    if len(st.session_state['currently_picked'])==0:
        del st.session_state['currently_picked']
    else: 
        st.session_state['currently_picked'].pop()
        for i,j in enumerate(st.session_state['currently_picked']):
            if picks_order[i]==0:
                col6.image(".\\heroes\\"+hero_df[hero_df['sid']==j]['image_names'].values[0])
            else:
                col7.image(".\\heroes\\"+hero_df[hero_df['sid']==j]['image_names'].values[0])

#if predict is clicked execute this
if predict:
    
    name_list=[]
    #appending list with hero names 
    for j in st.session_state['currently_picked']:
        name_list.append(hero_df[hero_df['sid']==j]['localized_name'].values[0])
    
    #Predictions is a function defined in model_funcs_opp file which will take name filled array as input and give out 3 suggestions
    # as output
    ans=predictions(np.array(name_list))
    
    for i,j in enumerate(st.session_state['currently_picked']):
        if picks_order[i]==0:
            col6.image(".\\heroes\\"+hero_df[hero_df['sid']==j]['image_names'].values[0])
        else:
            col7.image(".\\heroes\\"+hero_df[hero_df['sid']==j]['image_names'].values[0])
    for i in range(3):
        col6.write('\n')
    col6.subheader('Predictions')
    #this block of code draws images of predictions onto the screen
    for i in range(3):
        col6.write('\n')
    for _ in ans:
        col6.image(".\\heroes\\"+hero_df[hero_df['sid']==_]['image_names'].values[0])
    #empties the dictionary on reload
    st.session_state['currently_picked']=[]

# st.write(st.session_state['currently_picked'])














