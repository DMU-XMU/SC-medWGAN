###===###
# Import some dependencies
import  numpy                   as      np
import  pandas                  as      pd
from    sklearn.preprocessing   import  MinMaxScaler
import  matplotlib.pyplot       as      plt

import 	torch

###===###
# Download our synthetic sepsis dataset from PhysioNet
# and place it in the A000_Inputs folder
Folder  = "./A000_Inputs/"
File    = "xxx.csv"

###===###
# Read in the synthetic sepsis dataset and treat it as the ground truth
MyData = pd.read_csv(Folder+File)
#print(MyData.isin([np.inf, -np.inf]).sum(), "infinity values in the DataFrame.")
#print(MyData.isna().sum(), "NaN values in the DataFrame.")
# 替换无穷大和 NaN 值
#MyData.replace([np.inf, -np.inf], np.finfo(np.float64).max, inplace=True)
#MyData.fillna(0, inplace=True)


 
 

# Drop the unneeded Unnamed: 0 and Timepoints columns
#MyData = MyData.drop(["Unnamed: 0", "Timepoints"], axis = 1)

# Rename the columns for our sanity
# 	Admn: 	Administrative purposes
# 	Demo: 	Demographics
# 	Vitl: 	Vital signs/variables
# 	Labs: 	Lab results
# 	Flud: 	Fluid measurements
# 	Vent: 	Ventilation
ColNameSwap = {"Admn001_ID":         "Admn001_ID",
               "hemoglobin":        "Labs001_hemoglobin",
               "ferritin":          "Labs002_ferritin",
               "ret_count":         "Labs003_ret_count",
               "segmented_neutrophils": "Labs004_segs",
               "tibc":               "Labs005_tibc",
               "mcv":                "Labs006_mcv",
               "serum_iron":         "Labs007_serum_iron",
               "rbc":                "Labs008_rbc",
               "creatinine":         "Labs009_creatinine",
               "cholestrol":         "Labs010_cholestrol",
               "copper":             "Labs011_copper",
               "ethanol":            "Labs012_ethanol",
               "folate":             "Labs013_folate",
               "hematocrit":         "Labs014_hematocrit",
               "glucose":            "Labs015_glucose",
               "tsat":               "Labs016_tsat",
               "gender":             "Demo001_Gender",
               }

# Perform name swapping
MyData.rename(
    columns = {**ColNameSwap, **{v:k for k,v in ColNameSwap.items()}},
    inplace=True)

###===###
# Create A001_DataTypes.csv to document data property
MyData_Types = pd.DataFrame()

# Including
# 	index: 		--
# 	name:  		--
# 	type:  		Real/binary/categorical
# 	num_classes:	The amount of levels for each variable; fixed 1 for real
# 	embedding_size:	Projection dimension using soft-embeddings
# 	index_start: 	The first variable location in the concatenated features
# 	index_end: 	The pairing last location
MyData_Types["index"]           = []	
MyData_Types["name"]            = []
MyData_Types["type"]            = []
MyData_Types["num_classes"]     = [] 	
MyData_Types["embedding_size"]  = []
MyData_Types["include"]         = []
MyData_Types["index_start"]     = []
MyData_Types["index_end"]       = []

###===###
# Create called A002_MyData.csv to store a machine-readable ground-truth dataset
MyData_Transformed = pd.DataFrame()

# No transformation required for patient ID
MyData_Transformed["Admn001_ID"] = MyData["Admn001_ID"]

# Transformation procedure varies for 
# 	Flt: float
# 	Bin: binary
# 	Cat: categorical

#---
# There are 2 different types of flt variables
# 	N2: Those with Naturally Normal (N2) distributions
# 	LN: Those that can be Logged to become Normal (LN)
Flt_Variable_N2 = \
[   ]

Flt_Variable_LN = \
[   "Labs001_hemoglobin", "Labs002_ferritin","Labs003_ret_count","Labs004_segs", "Labs005_tibc","Labs006_mcv",
    "Labs007_serum_iron","Labs008_rbc","Labs009_creatinine","Labs010_cholestrol", "Labs011_copper",
    "Labs012_ethanol", "Labs013_folate", "Labs014_hematocrit", "Labs015_glucose","Labs016_tsat"
    ]

#---
# Bin variables
Bin_Variable = \
[   "Demo001_Gender"
    ]

#---
# We need to separately store some back-transform statistics for later use
A001_BTS_Float                  = {}
A001_BTS_Float["Name"]          = []
A001_BTS_Float["min_X0"]        = []
A001_BTS_Float["max_X1"]        = []
A001_BTS_Float["LogNormal"]     = []

A001_BTS_nonFloat               = {}
A001_BTS_nonFloat["Name"]       = []
A001_BTS_nonFloat["Type"]       = []
A001_BTS_nonFloat["Quantiles"]  = []

###===###
# Call the helper function
minmax_scaler = MinMaxScaler()

#---
# For every Flt-N2
for itr in range(len(Flt_Variable_N2)):
    
    #---
    # if this is the first variable
    if itr == 0:
 	# initialise row number and index number in the DataTypes csv
        Cur_Types_Row = 0
        Cur_Index_Row = 0

    # otherwise
    else:
 	# update the row counts
        Cur_Types_Row = list(MyData_Types["index_end"])[-1]

    #---
    # Grab the corresponding variable and numpify it
    Cur_Name = Flt_Variable_N2[itr]
    Cur_Val  = np.array(list(MyData[Cur_Name]))

    #---
    # then document its properties
    # note, 
    # 	num_classes: 	1
    #   embedding_size: 1
    MyData_Types = MyData_Types.\
                   append({"index":             Cur_Index_Row,
                           "name":              Cur_Name,
                           "type":              "real",
                           "num_classes":       1,
                           "embedding_size":    1,
                           "include":           True,
                           "index_start":       Cur_Types_Row,
                           "index_end":         Cur_Types_Row + 1
                           },
                          ignore_index = True
                          )

    #---
    # Document the back-transformation statistics
    # to be transformed into the range of [0, 1]
    A001_BTS_Float["Name"].append(Cur_Name)

    # re-focus the min value to 0
    Temp_Val = torch.tensor(Cur_Val).view(-1)
    A001_BTS_Float["min_X0"].append(Temp_Val.min().item())

    # re-scale the max value to 1
    Temp_Val = Temp_Val - Temp_Val.min()
    A001_BTS_Float["max_X1"].append(Temp_Val.max().item())

    # Flt-N2 do not need to be logged
    A001_BTS_Float["LogNormal"].append(False)

    #---
    # Save the transformed data in the MyData csv
    Cur_Val = minmax_scaler.fit_transform(Cur_Val.reshape(-1, 1))
    MyData_Transformed[Cur_Name] = Cur_Val

    # tic....tick!
    Cur_Index_Row += 1
#---
# Now iterate through every Flt-LN variables
#print("这是mudata",MyData)
for itr in range(len(Flt_Variable_LN)):

    if itr == 0:
        # initialise row number and index number in the DataTypes csv
        Cur_Types_Row = 0
        Cur_Index_Row = 0

    # otherwise
    else:
        # update the row counts
        Cur_Types_Row = list(MyData_Types["index_end"])[-1]

    #---
    Cur_Name = Flt_Variable_LN[itr]
    #print("这是循环",MyData[Cur_Name])
    #print("这是循环list",list(MyData[Cur_Name]))
    Cur_Val  = np.array(list(MyData[Cur_Name]))
    #print("这是循环Cur_Val",Cur_Val)
    # Logify the variable
    #Cur_Val  = np.log(Cur_Val + 1)
    
    #Cur_Val  = np.log(Cur_Val + 1)
    Cur_Val = Cur_Val + 1
    Cur_Val = np.log(np.clip(Cur_Val, 1e-20, None))
    #Cur_Val = np.log(10) / np.log(np.e)

    
    #print("这是循环Cur_Val1",type(Cur_Val),Cur_Val)

    #---
    # Note, 
    # 	num_classes: 	1
    # 	embedding_size: 1
    MyData_Types = MyData_Types.\
                   append({'index'          : Cur_Index_Row,
                           'name'           : Cur_Name,
                           'type'           : 'real',
                           'num_classes'    : 1,
                           'embedding_size' : 1,
                           'include'        : True,
                           'index_start'    : Cur_Types_Row,
                           'index_end'      : Cur_Types_Row + 1
                           },
                          ignore_index = True
                          )
    #---
    A001_BTS_Float["Name"].append(Cur_Name)
        
    Temp_Val = torch.tensor(Cur_Val).view(-1)
    A001_BTS_Float["min_X0"].append(Temp_Val.min().item())

    Temp_Val = Temp_Val - Temp_Val.min()
    A001_BTS_Float["max_X1"].append(Temp_Val.max().item())

    # Flag the variable as logged
    A001_BTS_Float["LogNormal"].append(True)
    #print(type(Cur_Val),"这是类型")
    #print(Cur_Val,"这是数据")
     
    #---
    #max_value = np.finfo(np.float64).max
    #Cur_Val = np.clip(Cur_Val, -max_value, max_value)

    Cur_Val = minmax_scaler.fit_transform(Cur_Val.reshape(-1, 1))
    MyData_Transformed[Cur_Name] = Cur_Val
        
    # tic....tic....tick!!
    Cur_Index_Row += 1

#---
# Now iterate through all the Bin variables
for itr in range(len(Bin_Variable)):

    #---
    Cur_Types_Row = list(MyData_Types["index_end"])[-1]

    #---
    Cur_Name = Bin_Variable[itr]
    Cur_Val  = np.array(list(MyData[Cur_Name]))

    #---
    # Note,
    # 	num_classes: 	2
    # 	embedding_size: 2
    # 	index_end: 	Cur_Types_Row + 2
    MyData_Types = MyData_Types.\
                   append({"index":             Cur_Index_Row,
                           "name":              Cur_Name,
                           "type":              "bin",
                           "num_classes":       2,
                           "embedding_size":    2,
                           "include":           True,
                           "index_start":       Cur_Types_Row,
                           "index_end":         Cur_Types_Row + 2
                           },
                          ignore_index = True
                          )

    #---
    A001_BTS_nonFloat["Name"].append(Cur_Name)
    A001_BTS_nonFloat["Type"].append("bin")

    # Although Bin are non-numeric,
    # no qunatiles needed here
    A001_BTS_nonFloat["Quantiles"].append({})

    #---
    # Transform the non-numeric variables into a machine-readable version
    # For each availabel level (2 in the case for Bin)
    for itr2 in range(2):
        # Creates a column per level, and
        # suffixify the name with _1 or with _2
        Temp_Name = Cur_Name + '_' + str(itr2)
        
        # If originally of class 1, label 1 in _1, 0 otherwise
        # if originally of class 2, label 1 in _2, 0 otherwise
        Temp_Val  = np.zeros_like(Cur_Val)

	# Find the location of each levels
        Loc_Ele = np.where(Cur_Val == itr2)[0]
 	# Oneify the correct locations
        Temp_Val[Loc_Ele] = 1

	# Save the flagged locations of each level in the machine-readable dataset
        MyData_Transformed[Temp_Name] = Temp_Val
        MyData_Transformed[Temp_Name] = \
            (MyData_Transformed[Temp_Name]).astype(int)

    # tic....tic....tick**2!**3
    Cur_Index_Row += 1

###===###
# Recalibrate everything one last time for sanity checking
MyData_Types['index']           = (MyData_Types['index']).astype(         int)
MyData_Types['name']            = (MyData_Types['name']).astype(          str)
MyData_Types['type']            = (MyData_Types['type']).astype(          str)
MyData_Types['num_classes']     = (MyData_Types['num_classes']).astype(   int)
MyData_Types['embedding_size']  = (MyData_Types['embedding_size']).astype(int)
MyData_Types['include']         = (MyData_Types['include']).astype(       bool)
MyData_Types['index_start']     = (MyData_Types['index_start']).astype(   int)
MyData_Types['index_end']       = (MyData_Types['index_end']).astype(     int)

# Store the back-transformation statistics
BTS_Folder = "./Z001_Data1/BTS/"
torch.save(A001_BTS_Float,      BTS_Folder + 'A001_BTS_Float')
torch.save(A001_BTS_nonFloat,   BTS_Folder + 'A001_BTS_nonFloat')

# Store the variable description file
# and the machine-readable transformed ground-truth
Input_Folder = "./A000_Inputs/"
MyData_Types.to_csv(        Input_Folder + 'A001_DataTypes1.csv', index = False)
MyData_Transformed.to_csv(  Input_Folder + 'A002_MyData1.csv',    index = False)



















