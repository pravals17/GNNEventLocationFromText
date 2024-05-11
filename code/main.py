from utils import (haversineDistance, binary_search, has_loc_suffix, has_loc_domainwords, has_prep, get_wordshape, 
                   is_in_gazetteer, word2features, sent2features, sent2labels, sent2tokens, getLocationsGazPOSandSAPR, 
                   get_locs_NER, concat_placenames, get_NER_5WNER, getNodeFeatures, getNodeFeaturesWV, getPosFeatures, 
                   getTextualDistance, getCoToponyms, createDataObjects)
import pandas as pd
import numpy as np
import re
from math import radians, cos, sin, asin, sqrt
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
import joblib
from ast import literal_eval
from transformers import BertModel, BertTokenizer

import torch
from torch_geometric.data import InMemoryDataset, download_url 
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils.convert import from_networkx
from sklearn.metrics import accuracy_score 

class GNNBert(torch.nn.Module):
    def __init__(self):
        super(GNNBert, self).__init__()
        
        self.lin1 = nn.Linear(768,600)
        self.transfencod = nn.TransformerEncoderLayer(d_model=600, nhead=6)
        self.sage1 = SAGEConv(in_channels=600, out_channels=128, aggr="max")
        self.sage2 = SAGEConv(in_channels=128, out_channels=64, aggr="mean") 
        self.lin2 = nn.Linear(64,32)
        self.lin3 = nn.Linear(32,2)
        
    
    def forward(self, x, edge_index):
        h = self.lin1(x)
        h = torch.relu(h)
        h = self.transfencod(h)
        h = self.sage1(h, edge_index)
        h = torch.relu(h)
        h = self.sage2(h, edge_index)
        h = torch.relu(h)
        h = self.lin2(h)
        h = torch.relu(h)
        h = self.lin3(h)
        h = torch.relu(h)
        return F.log_softmax(h, dim=1)

def main():
    SANS = joblib.load('./NERFiles/SANS/crfNER895.pkl') #load NER system such as StanfordNER. Here, we use a custom NER, SANS.
    df_gaz = pd.read_csv('./Gazetteer/EnhancedHierarchicalGazetteer.csv', encoding="latin-1")
    df_gaz = df_gaz.drop_duplicates()
    gaz_placenames = df_gaz['Placename'].to_list()
    gaz_placenames = [str(place).strip() for place in gaz_placenames]
    gaz_placenames = [re.sub('[^A-Za-z0-9]+', ' ', place) for place in gaz_placenames]

    placeNames = gaz_placenames #for SANS to identify placenames
    placeNames.sort() # Used to create feature related to whether a word is present in the gazetteer the SANS
    df_hiergaz = pd.read_csv('./Gazetteer/HierarchicalGazetteerIndia.csv', encoding="latin-1")
    
    country = list(set(df_hiergaz['Country'].str.lower().to_list()))
    state = list(set(df_hiergaz['State'].str.lower().to_list()))
    state.sort()
    district = list(set(df_hiergaz['District'].str.lower().to_list()))
    district.sort()
    subdistrict = list(set(df_hiergaz['Subdistrict'].str.lower().to_list()))
    subdistrict.sort()
    placename = list(set(df_hiergaz['Placename'].str.lower().to_list()))
    placename = [str(plc) for plc in placename]
    placename.sort()
    df_train = pd.read_csv('./TrainValTestData/train.csv')
    df_test= pd.read_csv('./TrainValTestData/test.csv')
    data_list = []
    
    #load BERT tokenizer and model
    tokenizerBert = BertTokenizer.from_pretrained('bert-base-uncased')
    modelBert = BertModel.from_pretrained('bert-base-uncased')
    
    data_listTrain = createDataObjects(df_train, SANS, modelBert, tokenizerBert, country,state,district,subdistrict,placename)
    data_listTest = createDataObjects(df_test, SANS, modelBert, tokenizerBert)
    dataloader = DataLoader(data_listTrain, batch_size=32)
    batch = next(iter(dataloader))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNNBert().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=5e-4)
    epochs = 100
    lossdist = []
    model.train()
    embdist = []
    preds = []
    
    for epoch in range(epochs+1):
        total_loss = 0
        acc = 0
        for batch in dataloader:
            optimizer.zero_grad()
            out = model(batch.x.to(device), batch.edge_index.to(device))
            preds.append(out)
            loss = criterion(out, batch.y.to(device))
            lossdist.append(loss)
            total_loss += loss
            acc = accuracy_score(batch.y.cpu(), out.argmax(dim=1).cpu())
            loss.backward()
            optimizer.step()
        if(epoch % 10 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {total_loss/len(dataloader):.3f} | Train Acc:'f' {acc*100:>6.2f}%')
    
    #torch.save(model.state_dict(), './model.pth')
    
if __name__ == "__main__":
    main() 