import random; random.seed(53)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics

import pandas as pd
from datasets.helper_functions import plot_confusion_matrix
from datasets.helper_functions import plot_and_return_top_features

import streamlit as st 

def main():
    st.title("Who's Tweeting? Trump or Trudeau?")
    st.sidebar.title("Tweet Classification Web App")
    

    @st.cache(persist=True)
    def load_data():
        tweet_df = pd.read_csv('datasets/tweets.csv')
        return tweet_df

    @st.cache(persist=True)
    def split_data(ds):
        y = ds['author']
        X_train, X_test, y_train, y_test = train_test_split(ds.status, y, test_size=0.33, random_state=53)
        return X_train, X_test, y_train, y_test
    

    @st.cache(persist=True)
    def vactorize_tweets(X_train, X_test):
        tfidf_vectorizer = TfidfVectorizer(stop_words="english", min_df=0.05, max_df=0.9)
        tfidf_train = tfidf_vectorizer.fit_transform(X_train)
        tfidf_test = tfidf_vectorizer.transform(X_test)
        return tfidf_train, tfidf_test, tfidf_vectorizer

    @st.cache(persist=True)
    def train_model(tfidf_train, tfidf_test, y_train, y_test):
        tfidf_svc = LinearSVC()
        tfidf_svc.fit(tfidf_train, y_train)
        tfidf_svc_pred = tfidf_svc.predict(tfidf_test)
        tfidf_svc_score = metrics.accuracy_score(tfidf_svc_pred, y_test)
        svc_cm = metrics.confusion_matrix(y_test, tfidf_svc_pred)
        return tfidf_svc, tfidf_svc_pred, tfidf_svc_score, svc_cm

    def predict_tweet(tweet):
        tweet_vectorized = tfidf_vectorizer.transform([tweet])
        tweet_pred = tfidf_svc.predict(tweet_vectorized)
        return tweet_pred

    def display_result():
        if tweet == "":
            st.write("Please Write Tweet")
        else:    
            prediction = predict_tweet(tweet)
            accuracy = tfidf_svc_score * 100
            if prediction[0] == "Donald J. Trump":
                st.write("Tweet is written by :",prediction[0])
                st.write("Accuracy:", accuracy.round(2))
                st.image('images/dt.png',width=250)
            else:
                st.write("Tweet is written by :",prediction[0])
                st.write("Accuracy:", accuracy.round(2))
                st.image('images/jt.png',width=150)
    
    ds = load_data()
    X_train, X_test, y_train, y_test = split_data(ds)
    tfidf_train, tfidf_test, tfidf_vectorizer = vactorize_tweets(X_train, X_test)
    tfidf_svc, tfidf_svc_pred, tfidf_svc_score, svc_cm = train_model(tfidf_train, tfidf_test, y_train, y_test)


    st.sidebar.subheader("Write Tweet")   
    tweet = st.sidebar.text_area("Tweet")
    
    if st.sidebar.button("Classify", key="classify"):
        display_result()
    
                
    if st.sidebar.checkbox("Show Dataset", False):
        st.subheader("Tweets Dataset")
        st.write(ds)

    if st.sidebar.checkbox("Show Confusion Metrix", False):
        st.subheader("Confusion Metrix")
        plot_confusion_matrix(svc_cm, classes=['Donald J. Trump', 'Justin Trudeau'], title="LinearSVC Confusion Matrix")
        st.pyplot()

    if st.sidebar.checkbox("Introspect Model", False):
        plot_and_return_top_features(tfidf_svc, tfidf_vectorizer)
        st.pyplot()

    
    
if __name__ == '__main__':
    main()