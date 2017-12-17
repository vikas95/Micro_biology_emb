#from keras import padding
from keras.models import Sequential
import numpy as np
import keras
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Activation
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.layers import Merge
from keras.layers import Dropout
from keras.layers.wrappers import Bidirectional
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from keras.layers import Dense
from keras.layers import Flatten
from nltk.tokenize import RegexpTokenizer  ### for nltk word tokenization
tokenizer = RegexpTokenizer(r'\w+')


file1=open("Final_training_sents_042417.txt","r")
All_data=[]
Data=[]
Data_len=[]
tags=[]
prev_Data=[]
count=0
tag_count=0
for line in file1:
    count+=1

    Words=line.split()
    if Words[1:] == prev_Data:
       if str(Words[0])=="9.6": # or str(Words[0])=="7.2":
          tags[-1]=1
          tag_count+=1


    else:

        sent=" ".join(Words[1:])
        act_words=tokenizer.tokenize(sent)
        All_data += act_words
        Data.append(act_words)
        Data_len.append(len(act_words))
        # print(count)
        if str(Words[0])=="9.6": # or str(Words[0])=="7.2":
           tags.append(1)
           tag_count += 1
        else:
           tags.append(0)
    prev_Data=Words[1:]

    if tag_count==550:
       boundary=len(Data)

print("boundary is: ", boundary)
print("data len is: ", len(Data))
"""    
print (len(tags))
print(len(Data))

print (Data_len[0:10])
print (tags[0:10])
"""

maxlen_1=max(Data_len)
print (maxlen_1)
All_words=list(set(All_data))
print (len(All_words))

word2ind = {word: index + 1 for index, word in enumerate(All_words)}
ind2word = {index + 1: word for index, word in enumerate(All_words)}

max_words=max(word2ind.values())+1

X_enc = [[word2ind[c] for c in x] for x in Data]
X_enc=pad_sequences(X_enc,maxlen=maxlen_1)

X_train=X_enc[:boundary]
X_test=X_enc[boundary:]

y_train=tags[:boundary]
y_test=tags[boundary:]


model=Sequential()
model.add(Embedding(max_words, 100, input_length=maxlen_1))  #, mask_zero=True

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='ADAM', metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(X_train, y_train, epochs=5, batch_size=100, verbose=2)   ## , validation_data=(X_test, y_test)
# Final evaluation of the model
preds=model.predict_classes(X_test)

#scores = model.evaluate(X_test, y_test, verbose=0)


count=0
TP=0
FP=0
TN=0
FN=0
for pred1 in preds:
    if pred1==1:
       if y_test[count]==1:
          TP+=1
       else:
          FP+=1
    else:      ## when pred1=0
       if y_test[count] == 1:
          FN += 1
       else:
          TN += 1

    count+=1


precision=TP/float(TP+FP)

recall=TP/float(TP+FN)

print ("precision is: ",precision)
print ("recall is: ",recall)


"""
import os
import glob
import nltk
folder_name='Bergey_strict_filter/Bergey_Vol4_pages'
# out_path=os.mkdir('Bergey_strict_filter/Bergey_2B_pages_strict_sents')


# Negative_files=open(os.path.join(folder_name,"Negative_files.txt"),'w')
Negative_files=open("Negative_files_Vol4.txt","w")
Positive_files=open("Positive_files_Vol4.txt","w")

for file_name in glob.glob(os.path.join(folder_name, '*.txt')):
    input_data=[]
    input_lines=[]
    current_file=open(file_name,"r")
    para_line=""
    for line in current_file:

        if line!="\n":
           para_line=para_line+" "+ line

        else:
           test_lines=nltk.sent_tokenize(para_line)

           for line3 in test_lines:
               input_word = line3.split()
               input_data.append(input_word)
               input_lines.append(line3)
           para_line = ""

    test_enc_line = [[word2ind[c] for c in x if c in All_words] for x in input_data]
    test_enc_line = pad_sequences(test_enc_line, maxlen=maxlen_1)
    doc_preds = model.predict_classes(test_enc_line)
    pred_vals=[]
    for pred1 in doc_preds:
        pred_vals.append(pred1[0])
    if pred_vals.count(1) == 0: ### 0 positive sentences, delete files
       Negative_files.write(str(file_name)+"\n")
       os.remove(file_name)
    else:
       pos_sent_files=open(str(file_name)+"strict",'w')
       Positive_files.write(str(file_name) + "\n")
       for ind5, val5 in enumerate(pred_vals):
           if val5==1:
              if ind5!=0 and ind5<len(input_lines)-1:
                 pos_sent_files.write(input_lines[ind5-1] + "\n")
                 pos_sent_files.write(input_lines[ind5]+"\n")
                 pos_sent_files.write(input_lines[ind5 + 1] + "\n")
              else:
                 pos_sent_files.write(input_lines[ind5] + "\n")


"""

