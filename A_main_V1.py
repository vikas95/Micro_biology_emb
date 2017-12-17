import os
import glob
from bs4 import BeautifulSoup
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import operator
from nltk.stem import PorterStemmer
ps = PorterStemmer()

import io


All_tags=[]
All_text=""
Compute_score=[]
Library_score=[]
line_num=0
Genus_text="Genus_text_lines.txt"
for file in glob.glob("Genus_files/*.html"):
    #print (file)
    file1=open(file,'r') #,encoding="ISO-8859-1")
    soup1=BeautifulSoup(file1,"html.parser")  #.encode("utf-8")
    All_tags=[]
    for tag in soup1.find_all():
        All_tags.append(tag.name)
    #print (set(All_tags))


    text1=soup1.find_all("p")
    for i in text1:
        # print ()
        All_text=All_text+(i.text.strip())+ " "       # #### In version2, we will calculate normalized values of count of these classes
        #Genus_text.write(i.text.strip()+"\n") #, encoding="iso-8859-1") #, encoding="ISO-8859-1")

        with io.open(Genus_text, "a", encoding="utf-8") as f:
            f.write(i.text.strip()+"\n")
    """                                                    # because the amount of text on different univ pages are not the same.
    text2 = soup1.find_all("p")
    for i in text2:
        All_text = All_text + (i.text.strip()) + " "
    """

All_text_list=All_text # .split()

print (set(All_tags))

