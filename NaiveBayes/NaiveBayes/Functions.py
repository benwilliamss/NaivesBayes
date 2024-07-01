import pandas as pd
from Classes import Dataset, DataFrequencyStore
import Globals 

def KFold(test_split_range, dataDF): #Splits thte dataframe into folds for training and test data 
    Data = Dataset()
    testRangeStart = test_split_range[0]
    testRangeEnd = test_split_range[1] 
    Data.Test = dataDF.iloc[testRangeStart: testRangeEnd] #Test data
    Data.Training = dataDF.iloc[0 : testRangeStart] #Training Data (first part)
    Data.Training  = pd.concat([Data.Training, dataDF.iloc[testRangeEnd : len(dataDF.index)]], axis=0)# add Training data (second part) to first part
    return Data #returns object containing test and training data 


def TrainData(dataframe, dataStoreObj, desiredClass):  #
    for index, row in dataframe.iterrows(): #will need to influence the numebr of rows 
        if row[Globals.attributes[0]] == desiredClass: #result we care about(classA or classB)
            #Iterates through each column for that row
            for attribute in Globals.attributes[1:len(Globals.attributes)]: #start at 1 to avoid class value 
                 dataStoreObj.Attribute_Values_Dict[row[attribute]] +=1   #Each line is adding/updating the ( : win count)
            dataStoreObj.ClassFrequency +=1 #Count frequency of class value

def NaiveBayes(TestDataDF, ClassATrainingData, ClassBTrainingData):
    predictedResults = []
    ClassAPriorProb = (ClassATrainingData.ClassFrequency  / (ClassATrainingData.ClassFrequency   +  ClassBTrainingData.ClassFrequency)) #Prior Probability of P(ClassA)
    ClassBPriorProb = (ClassBTrainingData.ClassFrequency  /  (ClassATrainingData.ClassFrequency  +  ClassBTrainingData.ClassFrequency)) #Prior Probability we lose P(ClassB)
    for index, testDataRow in TestDataDF.iterrows():
        ClassAPostProb  = ClassAPriorProb  #Store Prior Probability P(ClassA)
        ClassBPostProb = ClassBPriorProb   #Store Prior Probability P(ClassB)
        for attribute in Globals.attributes[1:len(Globals.attributes)]: #skip class column, only dependent feature attributes 
            ClassAPostProb  *= ((ClassATrainingData.Attribute_Values_Dict.get(testDataRow[attribute]) / ClassATrainingData.ClassFrequency))  #Multiply each conditional probability for a win
            ClassBPostProb  *= ((ClassBTrainingData.Attribute_Values_Dict.get(testDataRow[attribute]) / ClassBTrainingData.ClassFrequency))  #Multiply each conditional probability for a loss       
        ClassAProb = (ClassAPostProb / (ClassAPostProb + ClassBPostProb)) #Normalisation of probabilities
        ClassBProb = (ClassBPostProb / (ClassAPostProb + ClassBPostProb))
        result = Globals.classA if (ClassAProb > ClassBProb) else Globals.classB if (ClassAProb < ClassBProb) else "Undetermined"# determine outcome from highest probability, undetermined if equal
        predictedResults.append(result) #store all results in array 
    return predictedResults #return array of results for this fold 



def PrintConfusionMatrix(PredictedToActualResultsDict, fold): #add class variables - gloabl optimal
        print('\n                Confusion Matrix -',fold, 'folds\n                     ',  Globals.classA,' |    ',Globals.classB )
        for key, value in PredictedToActualResultsDict.items():
            match key: 
                case ('Predicted-ClassA&ActualClassA'): 
                    print('Predicted',  Globals.classA, ' |     ', value,'  |', end='')
                
                case ("Predicted-ClassB&ActualClassA"):
                    print('Predicted',  Globals.classB, '|     ', value,'   |',end='')
                case _:
                    print("     ", value)

def LaplaceSmoothing(dataframe): 
    #Purpose is to populate the data with an alpha value --> avoid zero frequency problem/clean data 
    #Jayaswal, V. (2020, November 22). Laplace smoothing in Naïve Bayes algorithm. Medium. https://towardsdatascience.com/laplace-smoothing-in-na%C3%AFve-bayes-algorithm-9c237a8bdece
    alpha = 1 #value to prepopulate the data with 
    PrepopulatedDataObj =  DataFrequencyStore() #will store depenedent features with alpha value
    PrepopulatedDataObj.ClassFrequency = alpha #class number(1) * alpha
    for index, row in dataframe.iterrows():
        #iterate through each row value 
        for value in row[1:len(row)]: 
            if not value in PrepopulatedDataObj.Attribute_Values_Dict: # Not in the dictionary? 
                PrepopulatedDataObj.Attribute_Values_Dict[value] = alpha #add data with alpha value
    return PrepopulatedDataObj #return populated data object

def UpdateClassDictValues(TestDataDF, classResults , ClassResultsDict):
    for index in range(0, len(classResults)): #iterate through results array 
        rowClassValue = TestDataDF.iloc[index][0] #reduce lookups of class value in the row
        result   = classResults[index] #result naives predicted 
        ClassResultsDict["Predicted-ClassA&ActualClassA"] +=  1 if(result  == Globals.classA  and  classResults[index] == rowClassValue )   else 0 
        ClassResultsDict["Predicted-ClassB&ActualClassB"] +=  1 if(result  == Globals.classB  and  classResults[index] == rowClassValue )   else 0
        ClassResultsDict["Predicted-ClassA&ActualClassB"] +=  1 if(result  == Globals.classA  and  classResults[index] != rowClassValue )   else 0
        ClassResultsDict["Predicted-ClassB&ActualClassA"] +=  1 if(result  == Globals.classB  and  classResults[index] != rowClassValue )   else 0 

