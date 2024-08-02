import copy
import pandas as pd
import numpy as np


class DecisionTreeClassifier():
    
    def __init__(self,originaldata):
        self.originaldata =originaldata
    def fit(self,data,cur_node,depth,attr,label,parent_node_class=None): #attr will be a dict
        
        print("----------training----------")
        
        
        if len(np.unique(data[label])) == 1:
            node_class = np.unique(data[label])[0]
            #print(node_class)
            #print('-----------')
            return {'val':node_class}
        elif len(data[label]) == 0:
            #print("ayhay")
            return self.originaldata[label].value_counts().idxmax()
        elif len(attr) ==1:
            return {'val':parent_node_class}
        
        
        else:
            node_class = data[label].value_counts().idxmax()
            best_att,best_ratio,best_thres,best_idx = self.find_best_split_of_all(data,attr,label)
            temp_attr = copy.deepcopy(attr)
            #print('--------------')
            #print(best_att)
            #print(best_ratio)
            del temp_attr[best_att]
            #print('--------------------\n')
            #if(best_att == 'legs'):
             #   print(data[best_att])
            
            if(attr[best_att] == 'categorical'):
                categories = np.unique(data[best_att])
                #print('categories')
                #print(categories)
                
                cur_node = {'split_attribute': best_att,'type': attr[best_att],
                            'val':node_class, 'categories':categories
                            }
                for value in categories:
                    #sub_data = data.where(data[best_feature] == value).dropna()
                    sub_data = data.where(data[best_att] == value).dropna()
                    cur_node[value] = self.fit(sub_data,{},depth+1,temp_attr,label,node_class) 
                self.trees = cur_node
                return cur_node
                
            else:
                cur_node = {'split_attribute': best_att, 'index_col':best_idx,
                            'threshold':best_thres,'type': attr[best_att],
                            'val':node_class
                            }
                
                left = data.where(data[best_att]<=best_thres).dropna()
                right = data.where(data[best_att]>best_thres).dropna()
                cur_node['left'] = self.fit(left, {}, depth+1, temp_attr,label,node_class)
                cur_node['right'] = self.fit(right, {}, depth+1, temp_attr,label,node_class)
                #self.depth += 1 
                self.trees = cur_node
                return cur_node
                
            
        
    def find_best_split_of_all(self,data,attr,label):
        best_att = None
        best_ratio = -1
        best_thres = None
        best_idx = None
        prev_count = -1
        # print('-------------------')
        for att in attr:
            
            if(att != label):
                gain_ratio,threshold,idx,counts = self.find_best_split(data, att,attr[att],label)
                # print('\n')
                # print(att)
                # print(gain_ratio)
                # print('\n')
                # if(best_att == None):
                #     best_att = att
                #     best_ratio = gain_ratio
                #     best_thres = threshold
                #     best_idx = idx
                # else:
                if(gain_ratio>best_ratio):
                    best_att = att
                    best_ratio = gain_ratio
                    best_thres = threshold
                    best_idx = idx
                    #prev_count = counts
                # elif(gain_ratio==best_ratio and prev_count>counts ):
                #     best_att = att
                #     best_ratio = gain_ratio
                #     best_thres = threshold
                #     best_idx = idx
                #     prev_count = counts
                
        #print('-------------------')
        
        return best_att,best_ratio,best_thres,best_idx
        
            
    def find_best_split(self, data,att,att_type,label):
        #print(data[label])
        prev_entropy = self.get_entropy(data[label])
        counts = []
        if(att_type == 'categorical'):
            vals,counts= np.unique(data[att],return_counts=True)
            tot_entropy = 0
            for i in range(len(vals)):
                cat_lbl = data.where(data[att]==vals[i]).dropna()[label]
                entropy = self.get_entropy(cat_lbl)
                tot_entropy += (counts[i]/np.sum(counts))*entropy
            info_gain = prev_entropy - tot_entropy
            #print(info_gain)
            #print("---------------------------------\n")
            if(info_gain==0):
                return info_gain,None,None,-1
            split_info = self.get_entropy(data[att])
            #print('split_info')
            #print(split_info)
            
            gain_ratio = info_gain/split_info
            return gain_ratio,None,None,0
        else:
            #print('------------sadasdasd')
            split_points = list(np.unique(data[att]))
            
            if len(split_points) > 100:
                split_points = list(range(0,100))
                mi = min(data[att])
                ma = max(data[att])
                gap = 1.0*(ma - mi)/100.0
                for i in range(len(split_points)):
                    split_points[i] = mi + split_points[i]*gap
            #print(values)
            best = 0
            threshold = 0
            idx = None
            
            for i in range(len(split_points)-1):
                point = split_points[i]
                left = data[att] <= point
                right = data[att] > point
                counts = [len(data[att][left].dropna()), len(data[att][right].dropna())]
                # print(counts, val)
                entropy_left = self.get_entropy(data[label][left].dropna())
                entropy_right = self.get_entropy(data[label][right].dropna())
                tot_entropy = (counts[0]/np.sum(counts))*entropy_left + (counts[1]/np.sum(counts))*entropy_right
                info_gain = prev_entropy - tot_entropy
                if(info_gain==0):
                    continue
                split_info = self.get_split_info(counts)
                gain_ratio = info_gain/split_info
                
                if gain_ratio>best:
                    best = gain_ratio
                    idx = i
                    threshold = point
            return best,threshold,idx,0
            # print(best, idx)
            
    def get_entropy(self,target_col):
        _,counts = np.unique(target_col,return_counts = True)
        entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(counts))])
        return entropy

    def get_split_info(self,counts):
        split_info = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(counts))])
        return split_info

    
    
            
    def predict(self, test):
        tree = self.trees
        #print(test)
        x = test.drop(columns=['class'], axis = 1).to_dict(orient = "records")
        predicted = pd.DataFrame(columns=["predicted"])
        i=0
        for index, row in test.iterrows():
            predicted.loc[index,"predicted"] = self._get_prediction(x[i])
            i+=1
        
        return predicted
    
    def _get_prediction(self, row):
        cur_layer = self.trees
        while cur_layer.get('split_attribute'):
            
            
            if(cur_layer['type'] == 'categorical'):
                cat = cur_layer['categories']
                ayhay = True
                for val in cat:
                    if(row[cur_layer['split_attribute']] == val):
                        cur_layer = cur_layer[val]
                        ayhay = False
                        break
                if(ayhay):
                    return cur_layer.get('val')
            else:
                if row[cur_layer['split_attribute']] <= cur_layer['threshold']:
                    cur_layer = cur_layer['left']
                else:
                    cur_layer = cur_layer['right']
        else:
            return cur_layer.get('val')
