import copy
import pandas as pd
import Globals 
from Functions import NaiveBayes,  KFold, PrintConfusionMatrix, LaplaceSmoothing, TrainData, UpdateClassDictValues

def main ():
    Globals.classA           = input("Enter class A value (Default: Win): ").replace(" ","") or "Win"   #Take user class A value or use default value "Win" - replace used ot minimise error
    Globals.classB           = input("Enter class B value (Default: Loss): ").replace(" ","") or "Loss" #Take user class B value or use default value "Loss"
    fileName                 = input("Enter data file path (Default: ./StartingXI_WinLoss.ods) : ").replace(" ","") or "./StartingXI_WinLoss.ods"
    try: 
        DataDF                   =  pd.read_excel(fileName) #Stores team data in a data frame
        DatasetNumRows           =  len(DataDF.index)   #Used to dedeuce folds to try & split quantity
        Globals.attributes       =  DataDF.columns      #Used in multiple functions 
        CumulativeAccuracyScores =  0                   #Track the accuracy scores across different folds
        PrePopulatedDataFreqObj  =  LaplaceSmoothing(DataDF) #populate object with alpha values 
        K_FoldsUsed              =  0 #counts the number of folds we have completed
        for K_Folds in range(3,11): #try folds 3..10
            if not((DatasetNumRows % K_Folds) == 0): continue #skip fold  if the dataset cannot produce a rational dataset split 
      
            ClassResultsDict = { #Store predicition accuracies - resets on each new fold 
                        "Predicted-ClassA&ActualClassA"   : 0, #Correct predicition for class A
                        "Predicted-ClassA&ActualClassB"   : 0, #Wrong predicition for class A 
                        "Predicted-ClassB&ActualClassA"   : 0, #Wrong predicition for class B
                        "Predicted-ClassB&ActualClassB"   : 0  #Correct predicition for class B
                    }
            splitQuantity  = int(DatasetNumRows / K_Folds) # Segment size for training and test data 
            testSplitRange = [0, splitQuantity]            #Test data slice - used for test set/fold range 
       
            for iteration in range(0, K_Folds):            #iterate through each fold   
                NaivesDatasets = KFold(test_split_range = testSplitRange, dataDF = DataDF) #produces object with training and test data
                ClassADataObj = copy.deepcopy(PrePopulatedDataFreqObj)                     #assign a copy of the object with prepopulated values - resets object each iteration
                ClassBDataObj = copy.deepcopy(PrePopulatedDataFreqObj) 
                TrainData(dataframe = NaivesDatasets.Training, dataStoreObj = ClassADataObj, desiredClass = Globals.classA) #Populate Class A training data Obj
                TrainData(dataframe = NaivesDatasets.Training, dataStoreObj = ClassBDataObj, desiredClass = Globals.classB) #Populate Class B training data Obj
                ClassPredictedResults = NaiveBayes(NaivesDatasets.Test, ClassADataObj, ClassBDataObj) #performs naive bayes returns predictions for that test data in that fold
                UpdateClassDictValues(TestDataDF=NaivesDatasets.Test, classResults = ClassPredictedResults, ClassResultsDict = ClassResultsDict) #update stored class predicitions
                testSplitRange[0]+= splitQuantity
                testSplitRange[1]+= splitQuantity 

            PrintConfusionMatrix(ClassResultsDict, K_Folds)
            CorrectPredictions   = (ClassResultsDict["Predicted-ClassA&ActualClassA"]  +  ClassResultsDict["Predicted-ClassB&ActualClassB"])
            IncorrectPredictions = (ClassResultsDict["Predicted-ClassB&ActualClassA"]  +  ClassResultsDict["Predicted-ClassA&ActualClassB" ]) 
            try: 
                AccuracyScore = (CorrectPredictions / (IncorrectPredictions + CorrectPredictions)) #Used to print out to console mean accuracy score
                CumulativeAccuracyScores+=AccuracyScore #stores all accuracy stores
                print("\nThis means that out of",(splitQuantity  * K_Folds), "test inputs naives predicted", CorrectPredictions, "correctly and with ", IncorrectPredictions, "incorrect predicitions.")
                print("This gives us a mean accuracy score for our NaivesBayes implementation of", (AccuracyScore*100),"percent.")
                K_FoldsUsed+=1
            except ZeroDivisionError:
                print("No mean produced as correct predicitions is",CorrectPredictions, "and incorrect predicitons is",IncorrectPredictions,"resulting in no Mean value produced.")
        try:
            print("\nOverall mean accuracy score across multiple K-Folds is...", ((CumulativeAccuracyScores)/K_FoldsUsed)*100) #Print mean accuracy scores across folds
        except ZeroDivisionError:
            print("The mean cumulative accuracy score",(CumulativeAccuracyScores),"cannot be divided by the number of kfolds used",K_FoldsUsed)
    except: 
        print("Incorrect file name entered.")
        print("Please restart the program and try again.")
    
    
main() #Program entry
