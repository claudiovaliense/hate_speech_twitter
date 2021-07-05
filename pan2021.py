#Author: Claudio Moises Valiense de Andrade. Licence: MIT. Objective: Soluction PAN 2021
import xml.etree.ElementTree as ET
import random
import sys
import joblib
import os
import io
import re

#"""
from sklearn.model_selection import StratifiedKFold
import claudio_funcoes as cv
def create_format_dataset():
    '''Transform dataset in format stratified'''
    texts = open("dataset/pan21-author-profiling-training-2021-03-14/orig/texts.txt", "w")
    scores = open("dataset/pan21-author-profiling-training-2021-03-14/orig/score.txt", "w")   
    label = ""    
    for filename in cv.list_files("dataset/pan21-author-profiling-training-2021-03-14/en/"):
        filename = f"dataset/pan21-author-profiling-training-2021-03-14/en/{filename}"
        if ".xml" in filename:
            tree = ET.parse(filename)            
            for child in tree.getroot().iter():        
                if child.tag == 'author': label = child.attrib['class']
                if child.tag == 'document':
                    texts.write(child.text + '\n')
                    scores.write(label+"\n")
    texts.close(); scores.close()
    cv.split_stratified_save_ids('pan21-author-profiling-training-2021-03-14')

def create_format_dataset2():
    '''Transform dataset in format stratified'''
    #name_file = 'pan21-author-profiling-test-without-gold-for-participants'
    name_file = 'pan21-author-profiling-training-2021-03-142'
    lang='en'
    texts = open(f"dataset/{name_file}{lang}/orig/texts.txt", "w")
    scores = open(f"dataset/{name_file}{lang}/orig/score.txt", "w")   
    label = ""   

    #scores_list = [score.strip() for score in open(f'dataset/pan21-author-profiling-training-2021-03-142{lang}/orig/score.txt')]
    #for label in scores_list:
    #    scores.write(label+"\n") 

    #for doc_train in open(f'dataset/pan21-author-profiling-training-2021-03-142{lang}/orig/texts.txt'):
    #    texts.write(doc_train)

    for filename in cv.list_files(f"dataset/{name_file}/{lang}/"):
        filename = f"dataset/{name_file}/{lang}/{filename}"
         
        document = []
        if ".xml" in filename:
            contents =  io.open(filename, 'r', encoding='utf-8', errors='ignore').read()
            tree = ET.fromstring(contents)  
            #tree = ET.parse(filename)            
            #for child in tree.getroot().iter():        
            for child in tree.iter():        
                if child.tag == 'author': label = child.attrib['class']
                if child.tag == 'document':
                    document.append(child.text )
            texts.write(" ".join(document) +"\n")
            scores.write(label+"\n")
    
    texts.close()
    #for score in range(100): ## add escore para a parte de teste, nao e validade esse escore
    #    scores.write("99\n") 

    #split = open(f'dataset/pan21-author-profiling-test-without-gold-for-participants{lang}/split_5.csv', 'w')
    #split.write(" ".join(map(str, list(range(200)) )) +';' +" ".join( map(str, list(range(200, 300)) )))
    #split.close()
    scores.close()
    


    cv.split_stratified_save_ids(f'{name_file}{lang}')
#"""

def list_files(dir):
    """Return list of the files in directory. Example: list_files('Downloads'); Return=['a.txt', 'b.txt']"""
    files_name =[]
    for r, d, files_array in os.walk(dir):
        for f in files_array:
            files_name.append(f)
    return files_name

def evaluate_pan2021():      
    #name_dataset='pan21-author-profiling-training-2021-03-142'; 
    name_dataset=sys.argv[2]; 
    classifier='svm'; r = "word_tfidf_bigram"
    vectorizer =  joblib.load(f"save_model/pan21-author-profiling-training-2021-03-142es_{r}")
    estimator = joblib.load(f"save_model/pan21-author-profiling-training-2021-03-142es_{r}_{classifier}") # load estimator
    documents = []; filenames = []
    
    escape_illegal_xml_characters = lambda x: re.sub(u'[\x00-\x08\x0b\x0c\x0e-\x1F\uD800-\uDFFF\uFFFE\uFFFF]', '', x)
    #for filename in list_files("dataset/{name_dataset}/en/"):
    for filename in list_files(f"{name_dataset}/es"):
        #filename = f"dataset/{name_dataset}/en/{filename}"        
        filename = f"{name_dataset}/es/{filename}"        
        #print(filename)
        document = []
        if ".xml" in filename:
            filenames.append( filename )            
            contents =  io.open(filename, 'r', encoding='utf-8', errors='ignore').read()
            tree = ET.fromstring(contents)                        
            #tree = ET.parse(filename)
            #for child in tree.getroot().iter():                    
            for child in tree.iter():                                                
                if child.tag == 'document':
                    document.append(child.text )
                    #print(document); exit()
            documents.append( " ".join(document) )
    x_test = vectorizer.transform( documents )
    y_pred = estimator.predict(x_test)
    

    index=0
    #for filename in list_files("dataset/{dataset}/en/"):
    for filename in list_files(f"{name_dataset}/es"):
        if ".xml" in filename:
            write = open(f"{sys.argv[4]}/{filename}", 'w')
            write.write(f"<author id=\"{ filename.split('.')[0] }\" lang=\"es\" type=\"{ int( y_pred[index] ) }\" />")
            index+=1
    #print(y_pred)
    #print( [int(y.strip()) for y in open("dataset/pan21-author-profiling-training-2021-03-142/orig/score.txt")] )
        

if __name__ == '__main__':        
    #create_format_dataset()
    create_format_dataset2()
    #evaluate_pan2021()