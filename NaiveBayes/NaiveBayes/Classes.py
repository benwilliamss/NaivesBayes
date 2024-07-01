import pandas as pd
class Dataset:
    def __init__(self):
        self.Training = pd.DataFrame() #stores training data
        self.Test     = pd.DataFrame() #stores test data 

class DataFrequencyStore:  #used to group together information
    def __init__(self):
        self.ClassFrequency  = 0 
        self.Attribute_Values_Dict  = {}
