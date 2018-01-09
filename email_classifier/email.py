import os
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression

def make_Dictionary(train_dir):
    subdirs=['ham','spam']
    emails=[]
    for subdir in subdirs:
        current_path=train_dir+subdir
        emails = emails+[os.path.join(current_path,f) for f in os.listdir(current_path)]

    pos=0
    dictionary = {}
    converted_train_emails = []
    ps = PorterStemmer()
    # lmtzr = WordNetLemmatizer()
    # with open('all_email_lmtzr.txt', "a") as fp:    
    for mail in emails:  
        with open(mail) as m:
            converted_context=[]
            context = m.read()
            context=context.decode('utf-8','ignore').encode("utf-8")
            tokenized_context = word_tokenize(context)
            for word in tokenized_context:
                if (word in stopwords.words('english')) or (word.isalpha() == False):
                    continue
                if word not in dictionary:
                    dictionary[word]=pos
                    pos+=1
                converted_context.append(ps.stem(word))
            converted_train_emails.append(converted_context)
                # every_whole_email=' '.join(converted_context)
                # fp.write(every_whole_email+"\n")

    # dictionary=set(all_words)
    # lmtzr = WordNetLemmatizer()
    
        # twice to test if it is verb (no need to consider adj, adv, prep)
        # new_item = lmtzr.lemmatize(item, 'v')
        # if new_item == item:
        #     new_item = lmtzr.lemmatize(item)
    # dictionary = dictionary.most_common(num_feature)
    return dictionary, converted_train_emails

def train_extract_features(mail_dir,train_data,num_feature): 

    features_matrix = np.zeros((len(train_data),num_feature))
    n_email = 0;
    ps = PorterStemmer()
    for email in train_data:
        for word in email:
            if word in dictionary:
                features_matrix[n_email,dictionary[word]]+=1
        # features_matrix[n_email]=features_matrix[n_email]/np.mean(features_matrix[n_email])
        n_email = n_email + 1     
    return features_matrix

def test_extract_features(mail_dir,num_feature): 
    files=[]
    files = files+[os.path.join(mail_dir,f) for f in sorted(os.listdir(mail_dir),key=lambda x: int(x.split('_')[2][0:-4]))]
    features_matrix = np.zeros((len(files),num_feature))
    n_email = 0;
    # lmtzr = WordNetLemmatizer()
    ps = PorterStemmer()
    for email in files:  
        with open(email) as m:
            context = m.read()
            context=context.decode('utf-8','ignore').encode("utf-8")
            tokenized_context = word_tokenize(context)
            for word in tokenized_context:
                if (word in stopwords.words('english')) or (word.isalpha() == False):
                    continue
                word=ps.stem(word)
                if word in dictionary:
                    features_matrix[n_email,dictionary[word]]+=1
        # features_matrix[n_email]=features_matrix[n_email]/np.mean(features_matrix[n_email])
        n_email = n_email + 1     
    return features_matrix

if __name__ == "__main__":

    # Create a dictionary of words with its frequency
    train_dir = './email_classification_data/train_data/'
    dictionary,pred_emails = make_Dictionary(train_dir)
    num_feature = len(dictionary)
    with open("dictionary.txt","a") as fp:
        for elem in dictionary:
            fp.write(elem+"\n")
    # print type(dictionary)
    print num_feature
    print "Dictionary is built!"
    # exit()
    # exit()

    # Prepare feature vectors per training mail and its labels

    train_labels = np.zeros(4372)
    train_labels[3107:4371] = 1
    train_matrix = train_extract_features(train_dir,pred_emails,num_feature)
    print "Features are extracted!"

    # Training SVM and Naive bayes classifier
    logistic = LogisticRegression()
    logistic.fit(train_matrix,train_labels)
    # model1 = MultinomialNB()
    # model2 = LinearSVC()
    # model1.fit(train_matrix,train_labels)
    # model2.fit(train_matrix,train_labels)
    print "Classifiers are trained!"

    # Test the unseen mails for Spam
    test_dir = './email_classification_data/test_data/'
    test_matrix = test_extract_features(test_dir,num_feature)
    result = logistic.predict(test_matrix)
    # result1 = model1.predict(test_matrix)
    # result2 = model2.predict(test_matrix)
    result=[int(a) for a in result]
    # result1=[int(a) for a in result1]
    # result2=[int(a) for a in result2]

    with open('log_result.txt', "a") as fp:
        for i in range(len(result)):
            fp.write(str(i+1)+","+str(result[i])+"\n")