# Who is tweeting? Trump or Trudeau
This is a machine learning classifier which predicts weather President Trump or Prime Minister Trudeau is tweeting!. This classifier predicts tweets based on single feature. Tweets classification falls in NLP classification. Tweets are shorter text. The Dataset contains 400 tweets. In this project CountVectorizer and TfidfVectorizer are used for text preprocessing. I have used two classifiers Multinomial Naive Bayes and LinearSVC for text classification.
# Dependencies
* scikit-learn   https://scikit-learn.org/stable/install.html
* pandas         https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html
* numpy          https://numpy.org/install/
* matplotlib     https://matplotlib.org/3.2.2/users/installing.html
* streamlit      https://docs.streamlit.io/en/stable/troubleshooting/clean-install.html
# How to run Project
First install all Dependancies then download/clone this repository. this repository contains two project files notebook and a web application of project.
To run notebook.ipynb open this file in jupyter notebook and run. To run this file go to the directory of project, open terminal (command line) and run this command "streamlit run web_app.py" which will host streamlit app on localhost, a link will be given. Open that link in browser.
