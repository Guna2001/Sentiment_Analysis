from asyncio.windows_events import NULL
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from io import StringIO
from trankit import Pipeline
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split 
from keras.models import load_model
import string
import numpy as np
   


stopwords=["ஒரு", 'என்று', 'மற்றும்', 'இந்த', 'இது', 'என்ற', 'கொண்டு', 'என்பது', 'பல', 'ஆகும்', 'அல்லது', 'அவர்', 'நான்', 'உள்ள', 'அந்த', 'இவர்', 'என', 'முதல்', 'என்ன', 'இருந்து', 'சில', 'என்', 'போன்ற', 'வேண்டும்', 'வந்து', 'இதன்', 'அது', 'அவன்', 'தான்', 'பலரும்', 'என்னும்', 'மேலும்', 'பின்னர்', 'கொண்ட', 'இருக்கும்', 'தனது', 'உள்ளது', 'போது', 'என்றும்', 'அதன்', 'தன்', 'பிறகு', 'அவர்கள்', 'வரை', 'அவள்', 'நீ', 'ஆகிய', 'இருந்தது', 'உள்ளன', 'வந்த', 'இருந்த', 'மிகவும்', 'இங்கு', 'மீது', 'ஓர்', 'இவை', 'இந்தக்', 'பற்றி', 'வரும்', 'வேறு', 'இரு', 'இதில்', 'போல்', 'இப்போது', 'அவரது', 'மட்டும்', 'இந்தப்', 'எனும்', 'மேல்', 'பின்', 'சேர்ந்த', 'ஆகியோர்', 'எனக்கு', 'இன்னும்', 'அந்தப்', 'அன்று', 'ஒரே', 'மிக', 'அங்கு', 'பல்வேறு', 'விட்டு', 'பெரும்', 'அதை', 'பற்றிய', 'உன்', 'அதிக', 'அந்தக்', 'பேர்', 'இதனால்', 'அவை', 'அதே', 'ஏன்', 'முறை', 'யார்', 'என்பதை', 'எல்லாம்', 'மட்டுமே', 'இங்கே', 'அங்கே', 'இடம்', 'இடத்தில்', 'அதில்', 'நாம்', 'அதற்கு', 'எனவே', 'பிற', 'சிறு', 'மற்ற', 'விட', 'எந்த', 'எனவும்', 'எனப்படும்', 'எனினும்', 'அடுத்த', 'இதனை', 'இதை', 'கொள்ள', 'இந்தத்', 'இதற்கு', 'அதனால்', 'தவிர', 'போல', 'வரையில்', 'சற்று', 'எனக்']

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
position:relative;
display:flex;
justify-content:center;
align-elements: center;
height:100vh;
background-image:url("https://rekearney.files.wordpress.com/2021/02/popcorn.jpg?w=1154");
background-size: cover;
background-repeat: no-repeat;
background-attachment: local;

}}
[data-testid="stText"] 
{{
background-color: rgb(86 23 23 / 57%);
border-radius:10px;
text-align:center;
border: 10px;
bottom:0px;
left:0px;
right:0px;
font-family:sans-serif;
font-size:20px;
}}
[data-testid="stVerticalBlock"] 
{{
bottom:0px;
left:0px;
right:0px;
}}

[data-testid="stHeader"] {{
    background-color: rgba(0, 0, 0, 0);
}}
[data-testid="stFooter"] {{
    background-color: rgba(0, 0, 0, 0);
}}
"""


model= load_model('model.h5')

p=Pipeline('tamil')

#Tokenize
i=0
count=0
def tokenize (data): 
    ls=[]
    data1 = p.tokenize(str(data),is_sent="True")
    for j in range(len(data1['tokens'])):
      temp = data1['tokens'][j]
      ls.append(str(temp['text'])) 
    data=ls
    return(punc(data))

#Punctuation Removal
def punc (data):
  ls=[]
  punc = string.punctuation
  for i in range(len(data)):
    temp=data[i]
    if temp not in punc:
      ls.append(temp)
  index=stop(ls)
  return(index)

  #stopwords removal
def stop (data):
  ls=[]
  for i in range(len(data)):
    temp=data[i]
    if temp not in stopwords:
      ls.append(temp)
  index=lemmatization(ls)
  return(index)

  #lemma
def lemmatization(data):
    alp=data
    temp=[]
    for AB in range(len(alp)):
      test=alp[AB]
      a=p.lemmatize(test)  
      b=a['sentences']
      c=b[0]
      d=c['tokens']
      e=d[0]
      try:
        temp.append(e['lemma'])
      except:
        f=e['expanded']
        for i in range(len(f)):
          g=f[i]
          temp.append(g['lemma'])

    data=temp
    print(data)
    index=encode_predict(data)
    return(index)


#Encoding
def encode_predict(data):
  data=" ".join(data)
  data=[data]
  tokenizer = Tokenizer(num_words=1000, split=' ') 
  print(data)
  tokenizer.fit_on_texts(data)
  X = tokenizer.texts_to_sequences(data)
  X = pad_sequences(X,maxlen=34)
  print(model.predict(X))
  index=np.argmax(model.predict(X))
  print(index)
  return(index)


def main():
    st.markdown(page_bg_img, unsafe_allow_html=True)
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Home","OneLiner","Whole_Review"],
            icons=["house","envelope","book"],
            menu_icon="cast",
            default_index=0,
        )
    if selected == "Home":
        st.title("Movie sentiment Analysis")
        st.header("Introduction :")
        st.text('''              Sentiment analysis is the use of natural language processing, text analysis, 
computational linguistics, and biometrics to systematically identify, extract, 
quantify, and study affective states and subjective information. Here, In this
Project, I used sentimental Analysis on tamil data where data contains the  
movie reviews in tamil.By Using Sentimental analysis,Interpret whether he 
reviewabout movie is positve or not and by we can classify the movie is a 
good movie or not.we can use this in a recommendation system in which a user 
have a pleasentexperience on using it.''')
        st.text('''Here we have two Options
    1.OneLiner - Cateogrize the movie based on Single Comment.
    2.Whole Review - Cateogrize the movie based on collective comments.''')

    elif selected == "OneLiner":
        st.title("Give the One Liner Of Movie")
        movie= st.text_input("Movie Name : ")
        sent = st.text_input("Movie Review (In Tamil) : ")
        print(type(sent))
        if st.button("Predict"):
          st.success(sent,icon="✅")
          result=tokenize(sent)
          print(result)
          if result == 0:
              st.write(movie)
              st.write("Pretty Average movie. One Time Watchable ")
          elif result == 1:
              st.write(movie)
              st.write("Great Movie. A treat to watch")
              st.balloons()
          elif result == 2:
              st.write(movie)
              st.write("Terrible Movie. Watch it at your own risk")
        
    elif selected == "Whole_Review":
          zero=0
          one=0
          two=0         
          tot_review=[]
          string_data=""
          st.title("Give the Movie Review")
          movie= st.text_input("Movie Name : ")
          file = st.file_uploader("Collective Movie Reviews :", accept_multiple_files=True)
          stringio=""
          if file is not NULL:
            for uploaded_file in file:
                 stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                 string_data = stringio.read()
            c_movie_review=string_data.split("\n")
            for count in range(len(c_movie_review)):
                st.write(c_movie_review[count])
                tot_review.append(tokenize(c_movie_review[count]))
            for tr in range(len(tot_review)):
                if tot_review[tr]==0:
                  zero+=1

                elif tot_review[tr]==1:
                  one+=1

                elif tot_review[tr]==2:
                  two+=1
            top_review1=[]
            top_review1.append(zero)
            top_review1.append(one)
            top_review1.append(two)
            if np.argmax(top_review1) == 0:
              st.write(movie)
              st.write("Pretty Average movie. One Time Watchable ")
            elif np.argmax(top_review1) == 1:
              st.write(movie)
              st.write("Great Movie. A treat to watch")
              st.balloons()
            elif np.argmax(top_review1) == 2:
              st.write(movie)
              st.write("Terrible Movie. Watch it at your own risk")


if __name__=='__main__':
     main()


