"""
Python routine to generate code when same equation is to be used
based on the value of Xi.

Eg. In BGK, velocity ID decides the values of various parameters. Although the 
number of equation is just one and only alpha varies. 
"""

import re
import glob
import sys
#from User_defined_function import *
#import User_defined_function as Udf



#------------------------------------------------------------------
def GetValVariableUsingRegexPassed(PassedVarName, Text, Regex):

    regexExpr = Regex.replace('VariableName', PassedVarName)
    pattern = re.compile(regexExpr)
    
    NumMatchFound = 0

    for match in pattern.finditer(Text):
        NumMatchFound += 1
        value = match.group(6)

    if NumMatchFound > 1:
        print 'Warning! Multiple value of ',PassedVarName,' found'
    if NumMatchFound == 1:
        return value
    if NumMatchFound <= 0:
        print 'Warning! Cannot find the value of',PassedVarName
#End of function definition GetValueofVariable
#------------------------------------------------------------------



#-----------------------------------------------------------------
# Function to read the contents of a file.
#-----------------------------------------------------------------
def ReadFile(fileName):
    text = []
    try:
        file = open(fileName, 'r')
    except IOError:
        print ('File ' + fileName + ' not Found')
        sys.exit()

    with file:
        text.append(file.read())
        file.close()
        return text
#-----------------------------------------------------------------



def WriteToFile(text, fileName):
    file = open(fileName, 'w')
    file.write(text)
    file.close()


def FindPositionStringText(String, Text):

    position = []
    regexExpr = r''+String
    #print regexExpr
    pattern = re.compile(regexExpr)
    numMatched = 0

    for match in pattern.finditer(Text):
        pos = match.start()
        position.append(pos)
        numMatched = numMatched + 1
  
    if(numMatched >= 1):
        #print 'Multiple ',String,' found in the text at positions',position
        return position

    #elif(numMatched==1):
        #print String,' found in the text at position',position
    #    return position
#End of FindPositionStringText


def Comment(line,Translated_Text):
    if len(line)==0:
        Translated_Text = Translated_Text + '\n'
    else:
        Translated_Text += '//' + line + '\n'
    return Translated_Text
#End of Function Comment.
    

def GetValueofVariable(VariableName, Text):
    regexExpr = r'( |\t|\n)*(\b'
    regexExpr = regexExpr + VariableName
    regexExpr = regexExpr + r'\b)( |\t|\n)*(=)( |\t|\n)*(\w+\.\w+)(;)' 
    pattern = re.compile(regexExpr)
    NumMatchFound = 0

    for match in pattern.finditer(Text):
        NumMatchFound += 1
        value = match.group(6)

    if NumMatchFound > 1:
        print 'Warning! Multiple value of ',VariableName,' found'
    if NumMatchFound == 1:
        return value
    if NumMatchFound <= 0:
        print 'Warning! Cannot find the value of',VariableName
#End of function definition GetValueofVariable
 

def Code(Line, Translated_Text):
    Translated_Text += Line + '\n'
    return Translated_Text
#End of function code defintion. 


def InsertForLoop(i, start, finish, Text):
    Text = Text + '\n'
    Line = 'for ( int '+i+'='+start+'; '+i+'<'+finish+'; '+i+'++ ){'
    return(Code(Line, Text))
    


#----------------------------------------------------------
# Routine to check number of arguments supplied to routines 
# such as CompoVeloIdx, SpaIdx etc. 
#----------------------------------------------------------

def CheckNumArgs(FunctionName, ArgsGiven):

    # To Do:- Modify this for 2D code. 
    # USe spacedim and subtract one number of arguements.
    if FunctionName == 'CompoVeloSpaIdx' and ArgsGiven!=5:
        print 'CompoVeloSpaIdx expects 5 Arguements but supplied =', ArgsGiven 

    if FunctionName == 'CompoVeloIdx' and ArgsGiven!=2:
        print 'CompoVeloIdx expects 2 Arguements but supplied =', ArgsGiven 

    if FunctionName == 'CompoMacroSpaIdx' and ArgsGiven!=5:
        print 'CompoMacroSpaIdx expects 5 Arguements but supplied =', ArgsGiven 

    if FunctionName == 'SpaIdx' and ArgsGiven!=3:
        print 'SpaIdx expects 3 Arguements but supplied =', ArgsGiven 

# End of routine to check number of arguements.


#----------------------------------------------------------
# Fucntion to parse arguements of Functions
# such as CompiVeloIdx, SpaIdx etc.
#----------------------------------------------------------

def ParseArguements(FunName, Args):
    
    ArgDic = {}
    if FunName == 'CompoVeloSpaIdx':
        ArgDic['CompoId'] = Args[0]
        ArgDic['VeloId'] = Args[1]
        ArgDic['RelSpaIdx_X'] = Args[2]
        ArgDic['RelSpaIdx_Y'] = Args[3]
        ArgDic['RelSpaIdx_Z'] = Args[4]

    if FunName == 'CompoMacroSpaIdx':
        ArgDic['CompoId'] = Args[0]
        ArgDic['MacroVarId'] = Args[1]
        ArgDic['RelSpaIdx_X'] = Args[2]
        ArgDic['RelSpaIdx_Y'] = Args[3]
        ArgDic['RelSpaIdx_Z'] = Args[4]
    
    if FunName == 'SpaIndex':
        ArgDic['RelSpaIdx_X'] = Args[0]
        ArgDic['RelSpaIdx_Y'] = Args[1]
        ArgDic['RelSpaIdx_Z'] = Args[2]

    if FunName == 'CompoVeloIdx':
        ArgDic['CompoId'] = Args[0]
        ArgDic['VeloId'] = Args[1]

    return ArgDic
#End of function to parse arguements.


#----------------------------------------------------------
# Routine to get variable name. 
# This information might be needed in the final code. 
#----------------------------------------------------------

def ParseText(Text, Positions, TypeVar, TextAfterParsing): 
    for pos in Positions:
        Item = {}

        #Which type of variable to search for such as 
        #Dist_f, Weights, Micro_Vel_, Macro_Vars.
        VarNameStartPos = pos + len(TypeVar)
        VarNameEndPos = Text.find('[', VarNameStartPos)
        VarName  = Text[VarNameStartPos:VarNameEndPos].strip()
        Item['VarName'] = VarName
        
        #Everything inside a square bracket from which information
        #has to be extracted.
        ArgStartPos = VarNameEndPos+1
        ArgEndPos = Text.find(']', VarNameStartPos)
        Arguement =  Text[ArgStartPos:ArgEndPos].strip()
        Item['Arguement'] = Arguement

        #The following two positions will be used to insert the 
        #genrated code at the write place in the translated text.
        Item['StartPosTextInsert'] = pos
        Item['EndPosTextInsert'] = ArgEndPos+1

        #print Text[pos:ArgEndPos+1]

        #Getting Routine name such as CompoVeloIdx, CompoMacroSpaIdx etc.
        FunStartPos = ArgStartPos
        FunEndPos = Text.find('(', FunStartPos)
        FunName = Text[FunStartPos: FunEndPos].strip()
        Item['Function'] = FunName


        #Getting Function Arguments which will help in generating code in HiLeMMs.
        FunArgsStart = FunEndPos + 1
        FunArgsEnd = Text.find(')',FunArgsStart)
        FunArgs = Text[FunArgsStart:FunArgsEnd].strip()
        Item['FunArgs'] = FunArgs
        #Parsed_Text.append([Item])
        

        FunArgs = FunArgs.split(',')
        NumArgsFun = len(FunArgs)
        CheckNumArgs(FunName, NumArgsFun)
        
        for i in range(NumArgsFun):
            FunArgs[i] = FunArgs[i].strip()

        Item['ParsedArgs'] = ParseArguements(FunName, FunArgs)
        #print Item
        #Parsed_Text.append(Item)
        TextAfterParsing.append(Item)
        #Parsed_Text.append(ParseArguements(FunName, FunArgs))
        #print Parsed_Text
        #print TextAfterParsing

# End of Routine to parse Text.
#---------------------------------------------------------------------


#----------------------------------------------
# Function to merge two dictionaries.
#----------------------------------------------
def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

# End of merge function.
#-----------------------------------------------


#----------------------------------------------------
# Function to generate code for the coordinates in 
# the user defined function.
#----------------------------------------------------

def GenCodeCoordinates(Parsed_Text):

    for i in range(0,len(Parsed_Text)):
        
        CodeCoord = {}
        FunName = Parsed_Text[i]['Function']
        
        if FunName == 'SpaIndex':
            
            VariableName = Parsed_Text[i]['VarName']
            RelPos_X = Parsed_Text[i]['ParsedArgs']['RelSpaIdx_X']
            RelPos_Y = Parsed_Text[i]['ParsedArgs']['RelSpaIdx_Y']
            RelPos_Z = Parsed_Text[i]['ParsedArgs']['RelSpaIdx_Z']

            if VariableName == 'X':
                CodeCoord['GenCode'] = 'coordinates[OPS_ACC_MD0(0,' + RelPos_X + ',' + RelPos_Y + ',' + RelPos_Z + ')]'

            elif VariableName == 'Y':
                CodeCoord['GenCode'] = 'coordinates[OPS_ACC_MD0(1,' + RelPos_X + ',' + RelPos_Y + ',' + RelPos_Z + ')]'

            elif VariableName == 'Z':
                CodeCoord['GenCode'] = 'coordinates[OPS_ACC_MD0(2,' + RelPos_X + ',' + RelPos_Y + ',' + RelPos_Z + ')]'

            else:
                print 'Variable Name for Spatial Coordinates could only be X, Y or Z'

        Parsed_Text[i] = merge_two_dicts(Parsed_Text[i], CodeCoord)

# End of function to gen coordinate code.
#----------------------------------------------------



#----------------------------------------------------
# Function to generate code for the Macroscopic 
# Variables in the user defined function.
#----------------------------------------------------

def GenCodeMacroVars(Parsed_Text):
    
    for i in range(0,len(Parsed_Text)):
        
        CodeMacroVars = {}
        FunName = Parsed_Text[i]['Function']
        
        if FunName == 'CompoMacroSpaIdx':
            
            MacroVarId = Parsed_Text[i]['ParsedArgs']['MacroVarId']
            ComponentId = Parsed_Text[i]['ParsedArgs']['CompoId']
            RelPos_X = Parsed_Text[i]['ParsedArgs']['RelSpaIdx_X']
            RelPos_Y = Parsed_Text[i]['ParsedArgs']['RelSpaIdx_Y']
            RelPos_Z = Parsed_Text[i]['ParsedArgs']['RelSpaIdx_Z']

            Index = 'VARIABLECOMPPOS[2 * ' +ComponentId+ '] + ' +MacroVarId
            CodeMacroVars['GenCode'] = 'macroVars[OPS_ACC_MD1(' + Index + ',' + RelPos_X + ',' + RelPos_Y + ',' + RelPos_Z + ')]'

        Parsed_Text[i] = merge_two_dicts(Parsed_Text[i], CodeMacroVars)

# End of function to gen code for macro vars.
#----------------------------------------------------



#----------------------------------------------------
# Function to generate code for the Microscopic Dist
# Fun in the user defined function.
#----------------------------------------------------

def GenCodeDistFun(Parsed_Text):

    for i in range(0,len(Parsed_Text)):
        
        CodeDistFun = {}
        FunName = Parsed_Text[i]['Function']
        
        if FunName == 'CompoVeloSpaIdx':
            
            ComponentId = Parsed_Text[i]['ParsedArgs']['CompoId']
            VeloId = Parsed_Text[i]['ParsedArgs']['VeloId']
            RelPos_X = Parsed_Text[i]['ParsedArgs']['RelSpaIdx_X']
            RelPos_Y = Parsed_Text[i]['ParsedArgs']['RelSpaIdx_Y']
            RelPos_Z = Parsed_Text[i]['ParsedArgs']['RelSpaIdx_Z']

            #ComponentNumber = 'Component' + ComponentId
            Index = 'COMPOINDEX[2 *' + ComponentId + '] + ' + VeloId
            CodeDistFun['GenCode'] = 'f[OPS_ACC_MD2(' +Index+ ',' + RelPos_X + ',' + RelPos_Y + ',' + RelPos_Z + ')]'
            
        Parsed_Text[i] = merge_two_dicts(Parsed_Text[i], CodeDistFun)

# End of function to gen code for distribution fun.
#--------------------------------------------------------------



#----------------------------------------------------
# Function to generate code for the Weights
#----------------------------------------------------

def GenCodeWeights(Parsed_Text):
    
    for i in range(0, len(Parsed_Text)):
        
        CodeWeights = {}
        FunName = Parsed_Text[i]['Function']
        VariableName = Parsed_Text[i]['VarName']

        if FunName == 'CompoVeloIdx' and VariableName=='':

            #print 'Running code gen for weights'
            
            ComponentId = Parsed_Text[i]['ParsedArgs']['CompoId']
            VeloId = Parsed_Text[i]['ParsedArgs']['VeloId']

            Index = 'COMPOINDEX[2 *' + ComponentId + '] + ' + VeloId
            CodeWeights['GenCode'] = 'WEIGHTS[' + Index+ ']'

        #print CodeWeights
        Parsed_Text[i] = merge_two_dicts(Parsed_Text[i], CodeWeights)

# End of function to gen code for Weights.
#--------------------------------------------------------------


#----------------------------------------------------
# Function to generate code for the Microscopic 
# velocity i.e. XI.
#----------------------------------------------------

def GenCodeXi(Parsed_Text):

    for i in range(0, len(Parsed_Text)):
        
        CodeXi = {}
        FunName = Parsed_Text[i]['Function']
        SpaceDim = int(UserVarsCpp['SpaceDim'])
        VariableName = Parsed_Text[i]['VarName']

        #print VariableName==''
        if FunName == 'CompoVeloIdx' and VariableName != '':

            #print FunName, VariableName
            
            ComponentId = Parsed_Text[i]['ParsedArgs']['CompoId']
            VeloId = Parsed_Text[i]['ParsedArgs']['VeloId']
            
            #ComponentNumber = 'Component' + ComponentId
            Index = 'COMPOINDEX[2 *' + ComponentId + '] + ' + VeloId

            if VariableName == 'Cx':    
                CodeXi['GenCode'] = 'XI[ (' +Index+ ') * LATTDIM]'
                    
            elif VariableName == 'Cy': 
                CodeXi['GenCode'] = 'XI[ (' +Index+ ') * LATTDIM +1]'

            elif VariableName == 'Cz':    
                CodeXi['GenCode'] = 'XI[ (' +Index+ ') * LATTDIM +2]'

            else:
                print 'Micro Velocity Variable Name should be Cx, Cy or Cz.'

        #print 'Hi',CodeXi
        Parsed_Text[i] = merge_two_dicts(Parsed_Text[i], CodeXi)

# End of function to gen code for Xi.
#--------------------------------------------------------------



#----------------------------------------------------
# Function to generate code for the Microscopic 
# velocity i.e. XI.
#----------------------------------------------------

def GenCodeXiRange(Parsed_Text):

    for i in range(0, len(Parsed_Text)):
        
        CodeXi = {}
        FunName = Parsed_Text[i]['Function']
        SpaceDim = int(UserVarsCpp['SpaceDim'])
        VariableName = Parsed_Text[i]['VarName']

        #print VariableName==''
        if FunName == 'CompoVeloIdx' and VariableName != '':

            #print FunName, VariableName
            
            ComponentId = Parsed_Text[i]['ParsedArgs']['CompoId']
            VeloId = Parsed_Text[i]['ParsedArgs']['VeloId']
            
            ComponentNumber = 'Component' + ComponentId
            
            if VariableName == 'Cx':    
                CodeXi['GenCode'] = 'XI[XiIdx * LATTDIM]'
                    
            elif VariableName == 'Cy':    
                CodeXi['GenCode'] = 'XI[XiIdx * LATTDIM + 1]'

            elif VariableName == 'Cz':    
                CodeXi['GenCode'] = 'XI[XiIdx * LATTDIM + 2]'

            else:
                print 'Micro Velocity Variable Name should be Cx, Cy or Cz.'

        #print 'Hi',CodeXi
        Parsed_Text[i] = merge_two_dicts(Parsed_Text[i], CodeXi)
        #print Parsed_Text[i],'\n'

# End of function to gen code for Xi.
#--------------------------------------------------------------


#-----------------------------------------------------------------------------
# Function to extract values from the user written 
# CPP file (Eg.- lbm3d_hilemms 

# FunName: function name which is constant as defined  by the iinterface.
# ArgNum : Number of arguement whose value is to be found. Note: Argname 
# might change and cannot be used directly. 
#------------------------------------------------------------------------------
def ParseCppFile(FileName, FunName, ArgNum):
    ArgValue = []
    CppText = ReadFile(FileName)[0]

    FunCallStartPos =  FindPositionStringText(FunName, CppText)[0]
   
    FunParanthesisStart = CppText.find('(', FunCallStartPos)

    FunParanthesisEnd = CppText.find(')', FunParanthesisStart)
    FunArgs = CppText[FunParanthesisStart+1:FunParanthesisEnd].strip()

    FunArgs = FunArgs.split(',')
    NumArgsFun = len(FunArgs)
        
    for i in range(NumArgsFun):
        FunArgs[i] = FunArgs[i].strip()

    ArgNameFindValue = FunArgs[ArgNum]
    ArgPosCppFile = FindPositionStringText(ArgNameFindValue, CppText)[0]
    ArgValStart = CppText.find('{', ArgPosCppFile)
    ArgValEnd = CppText.find('}', ArgValStart)
    
    if FunName == 'DefineComponents':
        String = CppText[ArgValStart+1:ArgValEnd].replace('"','')
        String = String.split(',')
        for i in range(0,len(String)):
            String[i] = String[i].strip()
        ArgValue.append(String)

    elif FunName == 'DefineMacroVars':
        String = CppText[ArgValStart+1:ArgValEnd].replace('"','')
        String = String.split(',')
        for i in range(0,len(String)):
            String[i] = String[i].strip()
        ArgValue.append(String)

    else:
        ArgValue.append(CppText[ArgValStart+1:ArgValEnd])
    #ArgValue[ArgNameFindValue] = CppText[ArgValStart+1:ArgValEnd]
    
    return ArgValue

# End of function to parse cpp file.
# ---------------------------------------------------



#------------------------------------------------------------------
# Function to parse information {x}^{m} to pow(x,m).
#------------------------------------------------------------------

def Parse_Base_Exponents(Text):
    regExpr= r'(\{)( )*([a-zA-Z0-9_\-,\[\]\(\) ]+)(\})( )*(\^)( )*(\{)( )*([0-9.]+)( )*(\})'
    pattern = re.compile(regExpr)
    StartPosExpr = []
    EndPosExpr = []

    Base = []
    Exponent = []
    
    #Loop to search base and exponent in user defined functions.
    for match in pattern.finditer(Text):

        #Base_Text saves the combination of matched groups in above
        #Re to get the exact base.
        Base_Text = ''
        for i in range(2, 4):
            if match.group(i) == None:
                Base_Text += ''
            else:
                Base_Text += match.group(i)
        
        StartIndex = match.start(0)
        EndIndex = match.end(0)
        #print Text[StartIndex:EndIndex]

        StartPosExpr.append(StartIndex)
        EndPosExpr.append(EndIndex)

        Base.append(Base_Text)
        Exponent.append(match.group(10))

    #print StartPosExpr
    # Insert new text pow(x,m) only if the pattern match was successfull. 
    if (StartPosExpr != []):

        #Loop to insert pow(base, exponent) at appropriate places.
        Temp = Text[:StartPosExpr[0]]
        Temp += 'pow(' + Base[0] + ',' + Exponent[0] + ')'

        for i in range(1,len(Base)):
            Temp += Text[EndPosExpr[i-1]+1:StartPosExpr[i]]
            Temp += 'pow(' + Base[i] + ',' + Exponent[i] + ')'

        Temp += Text[EndPosExpr[-1]+1:]
        Text = Temp

    #print Text
    return Text
# End of function to parse base and exponents.
#------------------------------------------------------------------



#----------------------------------------------------------------------
# 1. Function to Replace CalcBGKFeq(xiIndex, rho, u, v, w, T, polyOrder)
# with the user defined function.
#
# 2. Not accepting new function as its parameters may change (see code).
#
# 3. Filename is used to change and insert appropriate code in different file
# and may be removed later.
#---------------------------------------------------------------------- 
def ReplaceDistFunNewUdf(oldDistFunction, text, fileName):

    regexExpr = r'' + oldDistFunction
    pattern = re.compile(regexExpr)

    startPosOldDistFun = []
    endPosOldDistFun = []

    listFirstArgBGKFeq = []
    listLastArgBGKFeq = []

    numMatched = 0

    for match in pattern.finditer(text):
        if match != None:
            numMatched = numMatched + 1
            startPosFun = match.start(0)
            endPosFun = text.find(')', startPosFun)

            ParanthesisStart = text.find('(', startPosFun)
            ParanthesisEnd = endPosFun

            ArgsBGKFeq = text[ParanthesisStart + 1 :ParanthesisEnd].strip()
            ArgsBGKFeq = ArgsBGKFeq.split(',')
            
            FirstArgBGKFeq = ArgsBGKFeq[0]
            LastArgBGKFeq = ArgsBGKFeq[len(ArgsBGKFeq)-1]
            listFirstArgBGKFeq.append(FirstArgBGKFeq)
            listLastArgBGKFeq.append(LastArgBGKFeq)

            startPosOldDistFun.append(startPosFun)
            endPosOldDistFun.append(endPosFun + 1)
            
    if (numMatched >= 1):

        if (fileName == 'boundary_kernel.h'):
            userDefinedFunction = 'CalcUDFFeqNew(' + listFirstArgBGKFeq[0] + ', givenMacroVars, ' + listLastArgBGKFeq[0] + ')'
        else:
            userDefinedFunction = 'CalcUDFFeqNew(' + listFirstArgBGKFeq[0] + ', macroVars, ' + listLastArgBGKFeq[0] + ')'

        newText = text[:startPosOldDistFun[0]] + userDefinedFunction



        for ind in range(1, len(startPosOldDistFun)):
                 
            if (fileName == 'boundary_kernel.h'):
                userDefinedFunction = 'CalcUDFFeqNew(' + listFirstArgBGKFeq[ind] + ', givenMacroVars, ' + listLastArgBGKFeq[ind] + ')'
            else:
                userDefinedFunction = 'CalcUDFFeqNew(' + listFirstArgBGKFeq[ind] + ', macroVars, ' + listLastArgBGKFeq[ind] + ')'
            newText += (text[endPosOldDistFun[ind - 1]:startPosOldDistFun[ind]] + userDefinedFunction)

        newText += text[endPosOldDistFun[-1]:]
        return newText


    else:
        print 'No match found'
        return None

# End of function to replace old distribution function with 
# the user defined function.
#-----------------------------------------------------------



#-------------------------------------------------------------------------
# Function that will create UDF using text that has been translated.
# This function is just to complete the UDF function definition for C++. 
#-------------------------------------------------------------------------

def CreateUDF(Text):
    UDFFunction = 'Real CalcUDFFeqNew (const int XiIdx, const Real* macroVars, const int polyOrder)'
    UDFFunction += '\n{\n'
    UDFFunction += Text
    UDFFunction += '\nreturn result;\n' 
    UDFFunction += '\n}'  #End of function definition.  
    return (UDFFunction)

#End of CreateUDF function 
#--------------------------------------------------------



#-------------------------------------------------------------------------
# Function to insert the text before a string in a file.
# String acts a reference to find the correct pos for insertion. 
#-------------------------------------------------------------------------

def InsertTxtBeforeStringFile(FileName, TextToInsert, RefString):
    ReadText = ReadFile(FileName)[0]
    PosString = FindPositionStringText(RefString, ReadText)
    TextToWrite = ReadText[0:PosString[0]]
    TextToWrite += '\n\n' 
    TextToWrite += TextToInsert
    TextToWrite += '\n'
    TextToWrite += ReadText[PosString[0]:]
    WriteToFile(TextToWrite, FileName)

# End of function to insert UDF into model.h
#-------------------------------------------------------------------------


#-------------------------------------------------------------
# Warpper Function to insert UDF function call at 
# appropriate places in all kernel files.
#-------------------------------------------------------------

def InsertUDFFunctionCall():

    # Read and store all the data from cpp files.
    CppFiles = glob.glob("*kernel.h")

    CppTexts = []
    for File in CppFiles:
        CppTexts.append(ReadFile(File))

    # Global variable to store and replace the original CalcBGKFeq call.
    # First and last arg of CalcBGKFeq() needs to be separated.
    # First arg eg. = XiIdx, Last arg eg. =  polyOrder  
    #userDefinedFunction = ''

    for index in range(len(CppTexts)):

        print '--------------------------------------------------'
        print ' Working on File = ', CppFiles[index]
        print '--------------------------------------------------'

        oldFunctionCall = "CalcBGKFeq"
        udfCallInsertedText = ReplaceDistFunNewUdf(oldFunctionCall, CppTexts[index][0], CppFiles[index])

        if(udfCallInsertedText != None):
            WriteToFile(udfCallInsertedText, CppFiles[index])
    
# End of function to insert UDF function call.
#----------------------------------------------------------------------------------



FileName = 'Dist_eqn_3.txt'
#FileName = 'equation3.txt'
Text = ReadFile(FileName)[0]

#Parsed Text is the one where we are collecting information from user written file.
Parsed_Text = []

#This dictionary will hold values of variables like SpaceDim, Number of Components etc.
UserVarsCpp = {}

# Translated text would be the one where we will use
# Parsed text data and genarate the code that has to be inserted.
Translated_Text = ''

# Convert user written code for exponents into C++ format (i.e. insert pow(Variable, m) into code).
Text = Parse_Base_Exponents(Text)

#print 'Starting to Parse information from User written file'
#PlaceHolder = ['Dist_', 'Micro_Vel_', 'Weights', 'Macro_Vars', 'Coord_']
#PlaceHolder = ['Dist_', 'Micro_Vel_', 'Weights', 'Macro_Vars']
PlaceHolder = ['Macro_Vars', 'Weights', 'Micro_Vel_', 'Dist']

for String in PlaceHolder:
    Positions = FindPositionStringText(String, Text)
    ParseText(Text, Positions, String, Parsed_Text)
#print 'File parsing complete'

CppFileName = GetValueofVariable('CppFileName', Text)
UserVarsCpp['SpaceDim'] = ParseCppFile(CppFileName, 'DefineCase', 1)[0]

LatticeNames = ParseCppFile(CppFileName, 'DefineComponents', 2)[0]
#print LatticeNames

#Creating a dictionary of type C1:{LatNam:Val, Lattsize:Val}
#Need this info to generate code.
XiStart = 0
for i in range(0,len(LatticeNames)):
    CompDetails = {}
    CompNum = 'Component' + str(i)
    CompDetails['LattName'] = LatticeNames[i]
    CompDetails['LattSize'] = int(LatticeNames[i][3:])
    CompDetails['XiStart'] = XiStart
    CompDetails['XiEnd'] = XiStart + int(CompDetails['LattSize']) - 1
    XiStart = XiStart + int(CompDetails['LattSize'])
    UserVarsCpp[CompNum] = CompDetails


CompoIdMacroVars = ParseCppFile(CppFileName, 'DefineMacroVars', 3)[0]
NumComponents = len(LatticeNames)

MacroVarNames = ParseCppFile(CppFileName, 'DefineMacroVars', 1)[0]
#print MacroVarNames, CompoIdMacroVars

# Counting the starting and ending position of macroscopic variable of 
# each component. 
MacroVarStartPos = 0
for i in range(0, NumComponents):
    NumMacroVarsComp = 0

    for idx in range(0, len(CompoIdMacroVars)):
        if i == int(CompoIdMacroVars[idx]):
            NumMacroVarsComp += 1

    MacroVarEndPos = MacroVarStartPos + NumMacroVarsComp -1
    CompNum = 'Component' + str(i)
    UserVarsCpp[CompNum]['MacroStartPos'] = MacroVarStartPos
    UserVarsCpp[CompNum]['MacroEndPos'] = MacroVarEndPos
    MacroVarStartPos = MacroVarEndPos + 1



GenCodeCoordinates(Parsed_Text)
GenCodeDistFun(Parsed_Text)
GenCodeWeights(Parsed_Text)
GenCodeXi(Parsed_Text)
GenCodeMacroVars(Parsed_Text)

# for Val in Parsed_Text:
#     print Val

#Sorting the list according to the insertion position.
Parsed_Text_Sorted = sorted(Parsed_Text, key=lambda k: k['StartPosTextInsert'])
Translated_Text += Text[0:Parsed_Text_Sorted[0]['StartPosTextInsert']]

for i in range(0, len(Parsed_Text_Sorted)):
    Translated_Text = Translated_Text + Parsed_Text_Sorted[i]['GenCode']
    Start_Pos_Copy_Text = Parsed_Text_Sorted[i]['EndPosTextInsert']
    
    if i < len(Parsed_Text_Sorted)-1:
        End_Pos_Copy_Text = Parsed_Text_Sorted[i + 1]['StartPosTextInsert']
        Translated_Text += Text[Start_Pos_Copy_Text:End_Pos_Copy_Text]
    else:
        Translated_Text += Text[Start_Pos_Copy_Text :]

Translated_Text += '\n}' 
UDFFunction = CreateUDF(Translated_Text)

# FileToWrite = 'UDF_Translated.cpp'
# WriteToFile(Translated_Text, FileToWrite)

FileToWrite = 'UDF_Function.cpp'
WriteToFile(UDFFunction, FileToWrite)

"""
#UDF DECLRATION
FiletoInserDeclUDF = 'model.h'
UDFDecl = 'Real CalcUDFFeqNew(const int l, const Real* macroVars, const int polyOrder = 2);'
UDFDeclInsertBeforeText = '#endif' 
InsertTxtBeforeStringFile(FiletoInserDeclUDF, UDFDecl, UDFDeclInsertBeforeText)

#UDF DEFINITION
FileToWriteUDFDefinition = 'model.cpp'
UDFDefInsertBeforeText = '#include "model_kernel.h"'
InsertTxtBeforeStringFile(FileToWriteUDFDefinition, UDFFunction, UDFDefInsertBeforeText)
#InsertUDF(FileToWriteUDFDefinition, UDFFunction)

# UDF CALL
InsertUDFFunctionCall()
"""


"""
text = 'spacedim = 3;'
userRegex = r'( |\t|\n)*(\bVariableName\b)( |\t|\n)*(=)( |\t|\n)*([\w\.]+)(;)'
value = GetValVariableUsingRegexPassed('spacedim', text, userRegex)

text = r'{1:10}~{2|3|4}'
#text = r'{1:10}~{ }'

# Check to ensure that function didn't return None.
if(ParseRangeVariable(text)):
    MinRange, MaxRange, ListValNotIncluded = ParseRangeVariable(text)
    print MinRange, MaxRange, ListValNotIncluded

# text  = Udf.InsertForLoop('i', '1', '10', text)
"""