from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm  # Classifier SVN
from sklearn.naive_bayes import MultinomialNB  # Classifier MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier # Classifier ExtraTreesClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy # Manipulate array
import timeit  # Measure time
import sys # Import other directory
import claudio_funcoes_sub as cv  # Functions utils author
from sklearn.ensemble import RandomForestClassifier
#from bert_sklearn import BertClassifier
import xgboost as xgb 

""" Example execution: python3.6 name_dataset file_train file_test"""
SEED=42
numpy.random.seed(seed=SEED)
ini = timeit.default_timer() # Time process

def run_classifier(name_dataset, file_train, classifier, index, metric):
    """Run classifier"""            
    if classifier == "ada_boost":
        tuned_parameters =[{
            'n_estimators': [100],
            'learning_rate': [0.1]#,
            #'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1],
            #'algorithm': ['SAMME.R']
        }]
        tuned_parameters = [{}]
        estimator = AdaBoostClassifier(random_state=42, base_estimator= ComplementNB(alpha=0.01))

    elif classifier == 'bert':
        tuned_parameters = {'epochs':[3, 4], 'learning_rate' : [5e-5, 3e-5, 2e-5]}
        estimator = BertClassifier(validation_fraction=0, eval_batch_size=16, train_batch_size=16)
    
    elif classifier == "xgboost":
        #tuned_parameters = [{'learning_rate' : [0.001, 0.01, 0.1], 'gamma': [0.1, 1, 10]}]
        tuned_parameters = [{'booster': ['gbtree'], 'learning_rate' : [0.1, 1], 'gamma': [0.1, 1], 'n_estimators' : [100, 200] }
                ,{'booster': ['gblinear'], 'learning_rate' : [0.1, 1], 'n_estimators' : [100, 200] }] 
        #tuned_parameters = [{'learning_rate' : [0.1] }]
        estimator = xgb.XGBClassifier(random_state=SEED)

    elif classifier == "extra_tree":
        tuned_parameters =[{
            'n_estimators': [200],
            'max_features':['auto', None],
            'min_samples_split':[2,5,10],
            'min_samples_leaf':[1,5,10]
        }]
        estimator = ExtraTreesClassifier(random_state=SEED)
    
    elif classifier == "knn":
        tuned_parameters =[{
            #'n_neighbors': [2, 3]
            #'n_neighbors': [1, 2, 4, 8, 16, 32, 64, 128] #, 500, 1000],
            'n_neighbors': [1, 2, 4, 8, 16] #, 500, 1000],
            #'weights':['uniform', 'distance'],
            #'p':[1,2]
        }]
        estimator = KNeighborsClassifier()
    
    elif classifier == "logistic_regression":
        tuned_parameters =[{            
            #'C':[0.1],
            'C':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], #0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000
            #'solver' : ['sag' ]
            'max_iter':[1000]
        }]
        estimator = LogisticRegression(random_state=SEED)
    
    elif classifier == "naive_bayes":
        tuned_parameters =[{'alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10] }]
        estimator = MultinomialNB()
    
    elif classifier == "naive_bayes_complement":
        tuned_parameters =[{'alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10] }]
        estimator = ComplementNB()
    
    elif classifier == "passive_aggressive":
        tuned_parameters =[{
            #'C': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 1, 5, 10, 50, 100],         
            'C': [ 0.0001,  0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]#,
            #'max_iter' : [1000]
            #'loss': ['hinge', 'squared_hinge'],
            #'fit_intercept' : [False, True]            
        }]
        estimator = PassiveAggressiveClassifier(random_state=SEED, max_iter = 1000)
    elif classifier == 'decision_tree':

        tuned_parameters =[{   
            'min_samples_split': [2, 4, 8, 16, 32],
            'max_depth': [1, 2, 4, 8, 16, 32]
            #'min_samples_leaf': [1,2,4,8]
            #'class_weight': ['balanced', None]#,                    
        }]
        estimator = DecisionTreeClassifier(random_state=SEED)#, class_weight={0:1, 1:20})


    elif classifier == "random_forest":
        tuned_parameters =[{   
            'n_estimators': [10, 50, 100, 200, 500, 1000, 2000]#,
            #'max_depth': [2, 4, 8, 16, 32]
            #'min_samples_leaf': [1,2,4,8]
            #'class_weight': ['balanced']#,                    
        }]
        estimator = RandomForestClassifier(random_state=SEED)#, class_weight={0:1, 1:20})
        
    elif classifier == "sgd":
        #tuned_parameters =[{'alpha': [0.09419006441400779], 'eta0': [0.2573318908579897], 'learning_rate': ['optimal'], 'loss': ['perceptron']}]
        tuned_parameters = [{'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1], 'max_iter': [1000]}]#,
                             #"loss" : ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"]}]
        #tuned_parameters = [{'loss': ['log']}]
        '''tuned_parameters =[{
            'max_iter': [1000],
            'alpha': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1],
            "loss" : ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
            'penalty':['l1', 'l2'],
            'learning_rate': ['optimal', 'constant', 'invscaling', 'adaptive'],
            "eta0":[0.0001, 0.001, 0.01]
        }]'''
        estimator = SGDClassifier(random_state=SEED, max_iter=1000)

    elif classifier == "svm":        
        #tuned_parameters =[{'C': [3172.1647833335887], 'intercept_scaling': [0.10625908485515773], 'tol': [1.8668012261504774]}]
        #tuned_parameters = [{'C': [8174.20665483744], 'intercept_scaling': [1.6591603518373268], 'tol': [0.3634648237761219]}]
        #tuned_parameters = [{'C': [9628.352321918483], 'intercept_scaling': [0.12270878524431167], 'tol': [1875.1533903888399]}]
        #tuned_parameters = [{'C': [5884.365240442643], 'intercept_scaling': [0.1600737507295625], 'loss': ['hinge'], 'tol': [49415.29609756509]}]
        '''tuned_parameters =[
            {'kernel': ['rbf'], 'C': cv.C_flash, 'gamma': cv.gamma_flash},
            {'kernel': ['linear'], 'C': cv.C_flash}]'''
        #tuned_parameters = [{'kernel': ['rbf'], 'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'gamma': cv.gamma_flash}]
        #tuned_parameters = [{'kernel': ['rbf'], 'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'gamma' : [0.0001 ,0.001 ,0.01 , 0.1, 1, 10, 100 ] }]
        tuned_parameters = [{'C': [  0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}]#, 4000, 10000, 20000]} ]#, 100, 1000, 10000] }] , 'class_weight': ['balanced'] 
        #tuned_parameters = [{'C': [ 10000] }]

        #tuned_parameters = [{'kernel': ['rbf'], 'C': [ 0.1, 1, 10, 100]}]
        #tuned_parameters = [{'kernel': ['poly'], 'C': [ 0.1, 1, 10, 100, 1000]}]
        #tuned_parameters = [{'kernel': ['sigmoid'], 'C': [ 0.1, 1, 10, 100]}]
        #estimator = svm.SVC(**cv.svm_init_lbd)        
        #estimator = svm.SVC(random_state=SEED, max_iter=1000)
        #tuned_parameters = [{'C': [10000, 100000, 1000000], 'tol' : [1e-5]}]
        estimator = svm.LinearSVC(random_state=SEED, max_iter=1000)# dual=False)
        #estimator = svm.SVC(random_state=SEED, max_iter=1000, probability=True)#, kernel='linear')#, class_weight='balanced')

    elif classifier == "voting":
        tuned_parameters =[{
            'voting': ['hard']#, 'soft']            
        }]
        classifiers = [('naive_bayes', MultinomialNB()), ('naive_bayes_complement', ComplementNB())]
        estimator = VotingClassifier(estimators=classifiers)
    
    ''' # utilizar os tres melhores parametros do fold 0 para avaliar no grid para os demais
    if index != 0:
        escores = cv.load_escores(name_dataset, classifier, 0) # pega o escore do fold 0
        metric = 'f1_macro'
        tuned_parameters = []
        
        lista_melhores = cv.rank_grid(escores[0][metric], 3)
        #print(lista_melhores)
        for index_best in lista_melhores:
            k = list(escores[0][metric][index_best]['params'].keys())[0]            
            #print(escores[0][metric][index_best]['params'])
            tuned_parameters.append(escores[0][metric][index_best]['params'][k])               
        #print(tuned_parameters)
        tuned_parameters = [{k : tuned_parameters}]
    '''
        
    cv.escores_grid(name_dataset, classifier, file_train, estimator, tuned_parameters, [metric], 'f1_macro') # ['f1_macro', 'f1_micro'],  'f1_macro') #best param fold

name_dataset = sys.argv[1]
file_train = sys.argv[2]
classifier = sys.argv[3]
index = int(sys.argv[4]) #fold
metric = sys.argv[5] #fold
run_classifier(name_dataset, file_train, classifier, index, metric)
#time_dict = cv.load_dict_file('times/' +name_dataset +"_0")
#time_dict.update({'time_score' : (timeit.default_timer() - ini)})
print("Time End: %f" % (timeit.default_timer() - ini))
