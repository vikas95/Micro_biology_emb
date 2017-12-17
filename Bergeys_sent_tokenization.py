"""
file1=open("Bergey_sent.txt","r")
out_file=open("Prof_sents.txt","w")
import nltk
for line1 in file1:
    sents1=nltk.sent_tokenize(line1)
    for sentence1 in sents1:
        out_file.write("7.2"+" "+str(sentence1)+"\n")
"""
import os
import nltk
folder_name='Bergey_strict_filter/Bergey_2A_pages'
file1=open(os.path.join(folder_name,"14.txt"),"r").read()
para=file1.split("\n\n")
sent_num=0
for para1 in para:
    sents = ""
    for line1 in para1:
        sents+=str(line1)
    nltk_sen=nltk.sent_tokenize(sents)
    sent_num+=len(nltk_sen)

print(sent_num)