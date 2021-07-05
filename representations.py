#Author: Claudio Moises Valiense de Andrade. Licence: MIT. Objective: Create representations of texts
import os
import random
import sys  # Import other directory
import timeit  # Measure time
import numpy
import scipy
import torch
from afinn import Afinn
from scipy.sparse import hstack
from sklearn.datasets import dump_svmlight_file  # save format svmlight
from sklearn.datasets import load_svmlight_file, load_svmlight_files
from sklearn.decomposition import NMF  # modelagem de topicos
from sklearn.decomposition import (TruncatedSVD)
from sklearn.feature_extraction.text import TfidfVectorizer  # representation tfidf
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer,PowerTransformer, QuantileTransformer, StandardScaler)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import claudio_funcoes_sub as cv  # Functions utils author
random.seed(42); torch.manual_seed(42); numpy.random.seed(seed=42) # reproducibily soluction

def assign_GPU(Tokenizer_output):
    tokens_tensor = Tokenizer_output['input_ids'].to('cuda:0')    
    attention_mask = Tokenizer_output['attention_mask'].to('cuda:0')

    output = {'input_ids' : tokens_tensor,
          #'token_type_ids' : token_type_ids,
          'attention_mask' : attention_mask}

    return output

def vader(text):
    """Return score Vader"""
    analyzer = SentimentIntensityAnalyzer()  
    dict_vader = analyzer.polarity_scores(text)        
    return [dict_vader['neg'], dict_vader['neu'], dict_vader['pos']]
    #return [dict_vader['compound']]


def representation_bert(x, pooling=None):
    """Create representation BERT"""
    import numpy
    from transformers import BertModel, BertTokenizer
    
    if "16" in pooling: limit_token=16
    elif "32" in pooling: limit_token=32
    elif "64" in pooling: limit_token=64
    elif "128" in pooling: limit_token=128
    elif "256" in pooling: limit_token=256
    elif "512" in pooling: limit_token=512
    limit_token=512
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
    model = model.to('cuda:0') # gpu    
    for index_doc in range(len(x)):  

        inputs = tokenizer(x[index_doc], return_tensors="pt", max_length=limit_token, truncation=True) 
        inputs = assign_GPU(inputs)        
        outputs = model(**inputs)
        
        if 'bert_concat' in pooling or 'bert_sum' in pooling or 'bert_last_avg' in pooling or 'bert_cls' in pooling:
            hidden_states = outputs[2]
            token_embeddings = torch.stack(hidden_states, dim=0)
            token_embeddings = torch.squeeze(token_embeddings, dim=1) # remove a primeira dimensao que do embedding incial
            token_embeddings = token_embeddings.permute(1,0,2) # reordena para em cada linha ser um token diferente
            vets = []
            for token in token_embeddings:
                if 'bert_concat' == pooling:
                    vets.append( torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0).cpu().detach().numpy() ) # concatena as 4 ultimas dimensoes
                elif 'bert_sum' == pooling:
                    vets.append( torch.sum(token[-4:], dim=0).cpu().detach().numpy() )
                elif 'bert_last_avg' == pooling:
                    vets.append( torch.mean(token[-4:], dim=0).cpu().detach().numpy() )
                elif 'bert_cls' == pooling:
                    x[index_doc] = token[-1].cpu().detach().numpy() # o primeiro  token é o cls e ultima camada
                    break
            
            if 'bert_cls' != pooling:
                x[index_doc] = numpy.mean( vets, axis=0)
            
        else:
            tokens = outputs[0].cpu().detach().numpy()[0]            
            if 'bert_avg' in pooling:
                x[index_doc] = numpy.mean(tokens, axis=0) #average
            elif 'bert_max' in pooling: x[index_doc] = numpy.amax(tokens, axis=0)    
    return x

def process_transform(dataset, x_all=None):    
    for trans in sys.argv[6].split(','):
        if x_all == None:
            x_all = cv.file_to_corpus('dataset/' +dataset +'/orig/texts.txt')
            y = cv.file_to_corpus('dataset/' +dataset +'/orig/score.txt')
            y = [float(y) for y in y]        
        elif 'bert' in trans:
            x_all = [cv.preprocessor(x) for x in x_all]
            x = representation_bert(x_all, trans)
        elif trans == 'vader':            
            x_all = [cv.preprocessor(x) for x in x_all]
            x = [vader(x) for x in x_all]
        elif trans == 'affin':
            afinn = Afinn(emoticons=True)
            x = [[afinn.score(x)] for x in x_all]     
        try:
            os.mkdir("dataset/representations/"+dataset +'_f_' +trans) # Create directory
        except OSError:    
            print('directory exist')
        
        print(dataset +'_f_' +trans)
        dump_svmlight_file(x, y, "dataset/representations/" +dataset +'_f_' +trans +'/feature')
        cv.save_dict_file('times/' +name_dataset +"_0", {'time_representation': (timeit.default_timer() - ini)})

if __name__ == "__main__":
    ini = timeit.default_timer() # Time process
    name_dataset = sys.argv[1] # name dataset
    ids=sys.argv[2] # file ids
    datas=sys.argv[3] # file texts
    labels=sys.argv[4] # file label
    index = int(sys.argv[5]) # index fold
    rs = sys.argv[6]  # representations
    name_dataset_real = ids.split("/")[1]
    #process_transform(name_dataset_real) # generate bert

    try:    
        os.mkdir("dataset/representations/"+name_dataset) # Create directory
    except OSError:
        pass#; print('directory exist')
        
    x_train, y_train, x_test, y_test = cv.ids_train_test(ids, datas, labels, index)
    x_train = [cv.preprocessor(x) for x in x_train]
    x_test = [cv.preprocessor(x) for x in x_test]
    y_train = [float(y) for y in y_train] # float para permitir utilizar no classificador
    y_test = [float(y) for y in y_test]   

    if ',' in rs or 'bert' in rs:
        x_train = None
        x_test = None
        for r in rs.split(','):    
            scaler_boolean = False
            soma_um = False
            if '_min_max' in r:
                scaler_boolean = True
                scaler = MinMaxScaler()
                r = r.split('_min_max')[0]
            if '_scaler' in r:
                scaler_boolean = True
                if 'tfidf' in r: 
                    scaler = StandardScaler(with_mean=False)                        
                else : 
                    scaler = StandardScaler()                
                r = r.split('_scaler')[0]
            if '_maxabs' in r:
                scaler_boolean = True
                scaler = MaxAbsScaler()
                r = r.split('_maxabs')[0]
            if '_quantile' in r:
                scaler_boolean = True
                scaler = QuantileTransformer(random_state=42)
                r = r.split('_quantile')[0]
            if '_power' in r:
                scaler_boolean = True
                scaler = PowerTransformer()
                r = r.split('_power')[0]
            
            if '_normalizer' in r:
                scaler_boolean = True
                if '_normalizer_l1' in r:            
                    scaler =  Normalizer(norm='l1')
                    r = r.split('_normalizer_l1')[0]
                elif '_normalizer_l2' in r:            
                    scaler =  Normalizer(norm='l2')
                    r = r.split('_normalizer_l2')[0]
                elif '_normalizer_max' in r:            
                    scaler =  Normalizer(norm='max')
                    r = r.split('_normalizer_max')[0]                

            if 'word_tfidf' in r or 'char_tfidf' in r or 'graph' in r:# in r or 'roberta_min_max' in r or 'metafeature' in r:
                f_x_train, nao_usa1, f_x_test, nao_usa2 = load_svmlight_files([open('dataset/representations/' +name_dataset_real +'_' +r +'/train'+str(index), 'rb'), open('dataset/representations/' +name_dataset_real +'_' +r +'/test'+str(index), 'rb')])
            else:
                f_x, nao_usa = load_svmlight_file(open("dataset/representations/" +name_dataset_real +'_f_' +r +'/feature', 'rb'))    
                f_x_train, f_x_test = cv.ids_train_test_representation(ids, f_x, index, ' ')

            if scaler_boolean == True:                                     
                f_x_train = scaler.fit_transform(f_x_train)
                f_x_test = scaler.transform(f_x_test)
                
            if x_train is None:
                x_train = f_x_train
                x_test  = f_x_test 
            else:      
                x_train = hstack([ x_train, scipy.sparse.csr_matrix(f_x_train) ])
                x_test = hstack([ x_test, scipy.sparse.csr_matrix(f_x_test) ])        

        if ',' in rs or 'min_max' in rs or 'scaler' in rs or '_maxabs' in rs or '_porwer' in rs or '_normalizer' in rs:
            cv.save_dict_file('times/' +name_dataset +"_" +str(index), {'time_representation': (timeit.default_timer() - ini)})
        
        dump_svmlight_file(x_train, y_train, "dataset/representations/" + name_dataset +'/train'+str(index))
        dump_svmlight_file(x_test, y_test, "dataset/representations/" + name_dataset +'/test'+str(index))
        print("Time End: %f" % (timeit.default_timer() - ini))

    else: #utilizar para representacoes que dependem do fold
        r = rs
        if  'word_tfidf_bigram' in r:
            word_tfidf = TfidfVectorizer(ngram_range=(1,2))
            x_train = word_tfidf.fit_transform(x_train)    
            x_test = word_tfidf.transform(x_test)

        elif 'word_tfidf' in r :
            from sklearn.decomposition import PCA
            word_tfidf = TfidfVectorizer()#tokenizer=word_tokenize )
            x_train = word_tfidf.fit_transform(x_train)
            x_test = word_tfidf.transform(x_test)
            if 'pca' in r or 'svd' in r:
                lenght_reduction = int(sys.argv[7]) 
                if 'pca' in r:        
                    transformador = PCA(n_components=lenght_reduction, random_state=42)
                elif 'svd' in r:
                    print('SVD')
                    transformador = TruncatedSVD(n_components=lenght_reduction, random_state=42)
                x_train = transformador.fit_transform(x_train)
                #print(f"reduction: {x_train.shape}")
                x_test = transformador.transform(x_test)
                try:
                    os.mkdir( f"dataset/representations/{name_dataset}_{lenght_reduction}") # Create directory
                except:
                    pass
                dump_svmlight_file(x_train, y_train, f"dataset/representations/{name_dataset}_{lenght_reduction}/train{str(index)}")
                dump_svmlight_file(x_test, y_test, f"dataset/representations/{name_dataset}_{lenght_reduction}/test{str(index)}")

        elif r == 'char_tfidf':
            char_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,6))
            x_train = char_tfidf.fit_transform(x_train)
            x_test = char_tfidf.transform(x_test)

        elif r =='vader':
            x_train = [vader(x) for x in x_train]
            x_test = [vader(x) for x in x_test]

        elif r == 'affin':
            afinn = Afinn(emoticons=True)
            x_train = [[afinn.score(x)] for x in x_train]
            x_test = [[afinn.score(x)] for x in x_test]

        print("dataset/representations/" + name_dataset +'/train'+str(index))
        try:
            os.mkdir( f"dataset/representations/{name_dataset}") # Create directory
        except:
            pass
        dump_svmlight_file(x_train, y_train, f"dataset/representations/{name_dataset}/train{str(index)}")
        dump_svmlight_file(x_test, y_test, f"dataset/representations/{name_dataset}/test{str(index)}")
        cv.save_dict_file('times/' +name_dataset +"_" +str(index), {'time_representation': (timeit.default_timer() - ini)})

        print("Time End: %f" % (timeit.default_timer() - ini))
    