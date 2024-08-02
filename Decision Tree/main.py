import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from DecisionTreeClassifier2 import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def accuracy_score(y_true, y_pred):

	"""	score = (y_true - y_pred) / len(y_true) """

	return round(float(sum(y_pred == y_true))/float(len(y_true)) * 100 ,2)

def read_file_to_dict(filename):
    att_dict = {}
    f = open(filename, "r")
    i =0
    for line in f:
        words = line.split()
        att_dict[words[0]] = words[1]
    return att_dict

datasets = ['zoo.data','iris.data','wine.data','cmc.data','crx.data','german.data','krkopt.data','adult.data']
attributes = ['zoo_att.txt','iris_att.txt','wine_att.txt','cmc_att.txt','crx_att.txt','german_att.txt','krkopt_att.txt','adult_att.txt']

#datasets = ['adult.data']
#attributes = ['adult_att.txt']

open('../Files/result_decision.csv','w').close()
ct = 0
gap = '------------------------------\n------------------------------\n------------------------------\n'
for dataset_path,att_path in zip(datasets,attributes):
    label = 'class'
    att_full_path = "../datesets_classifier/"+att_path
    #print(att_full_path)
    attribute = read_file_to_dict(att_full_path)
    dataset_full_path = '../datesets_classifier/'+dataset_path
    print(dataset_path)
    df = pd.read_csv(dataset_full_path,
                          names=list(attribute.keys()))
    #df = df.sample(frac=1).reset_index(drop=True)
    
    
    #print(dataset)
    #print(attributes)
    
    cur_node = {}
    train,test = train_test_split(df, test_size=0.2)
    tree = DecisionTreeClassifier(train)
    tree.fit(train, cur_node, 0, attribute, label)
    #predictions = tree.predict(test)
    
    predictions = tree.predict(test)
    #print(predictions)
    print("-----------------\n")
    #print(test['class'])
    
    
    
    y_test = list(test['class'].to_dict().values())
    result = list(predictions['predicted'].to_dict().values())
    
    
    
    title = 'dataset,Accuracy %,Precision %,Recall %,f1 %\n'
    outf = open('../Files/result_decision.csv','a')
    if(ct==0):
        outf.write(title)
        ct+=1
    precision = 0
    recall = 0
    f1 = 0
    accuracy = (np.sum(predictions["predicted"] == test["class"])/len(test))*100
    print('Test Accuracy: {}'.format(accuracy))
    precision = precision_score(y_test, result, average='macro')*100
    print("Test Precision: {}".format(precision))
    recall = recall_score(y_test, result, average='macro')*100
    print("Test Recall: {}".format(recall))
    f1 = 2 * (precision * recall) / (precision + recall)
    print("Test F1 score: {}".format(f1))
    buffer_s = dataset_path.replace('.data','') + ',' + str(accuracy) + ',' + str(precision) +','+ str(recall) +','+ str(f1) + '\n'
    outf.write(buffer_s)
    outf.close()