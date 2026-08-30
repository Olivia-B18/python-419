"""
Olivia Buchanan
Homework 0: Python Practice
"""

# various variables
HAL = 9000
Bridge = "4"
Nahn = [2]
R2 = ["D", "2"]
C = [{"3PO"}]
Poppins = set("supercalifragilisticexpialidocious")
WALL = [["E"]]
K = {"2SO"}

# dictionaries
Galactica = {"Karl":"Helo",
             "Kara":"Starbuck",
             "Lee":"Apollo",
             }
Galactica["Sharon"] = "Boomer"
Galactica["Marge"] = "Racetrack"

Wheel = {"Blue": ["Moiraine", "Siuan"],
         "Green": ["Alanna"],
         "Purple": [],
         "White": ["Sarene"],
         "Yellow": [],
         "Plaid": [],
         "Red": ["Elaida", "Liandrin"],
         }

Wheel["Brown"] = ["Verin", "Saerin", "Janya"]
Wheel["Green"].append("Egweyne")
Wheel["Red"].remove("Liandrin")
Wheel["Yellow"].append("Nynaeve")
del Wheel["Plaid"]

# university levels
Germanna_levels = ["Freshman", "Sophomore"]
UMW_levels = ["Freshman", "Sophomore", "Junior", "Senior"]
MaryWash_levels = UMW_levels
CNU_levels = UMW_levels.copy()

def plus2(inputNum):
    """
    Returns the input number plus 2.
    """
    if type(inputNum) == int or type(inputNum) == float:
        return inputNum + 2

theSet = {"Navani", "Shallan", "Jasnah"}
def gimme_dat_set():
    """
    Returns the same set object per call.
    """
    datSet = theSet
    return datSet

def gimme_set_like_dat():
    """
    Returns a different set object per call
    """
    return {"Kaladin", "Dalinar", "Adolin"}

def center(inputString):
    """
    Returns the center letters of the input string.
    """

    if type(inputString) == str:
        strLen = len(inputString)

        if strLen == 0:
            return None
        elif strLen % 2 == 0:
            return inputString[(strLen // 2 - 1):(strLen // 2 + 1)]
        else:
            return inputString[strLen // 2]

def nuke_last(inputList, inputItem):
    """
    Removes the last element in the list with the 
    same value as the item.
    """

    if type(inputList) == list:
        inputList.reverse()
        inputList.remove(inputItem)
        inputList.reverse()

def middlest(inputNum1, inputNum2, inputNum3):
    """
    Returns the middle number of the given numbers.
    """

    listNums = [inputNum1, inputNum2, inputNum3]
    listNums.sort()
    return listNums[1]

def tack_on_end(inputList, inputVar, repeatCount=1):
    """
    Adds the value(s) in inputVar to inputList as many
    times as in repeatCount.
    """

    # need to iterate differently depending on type of inputVar
    if type(inputVar) == list:
        for _ in range(repeatCount):
            inputList.extend(inputVar)
    else:
        for _ in range(repeatCount):
            inputList.append(inputVar)

def wondrous_count(inputNum):
    """
    Returns the number of steps required to reach 1
    from the number in a Collatz sequence.
    """

    count = 0

    while (inputNum > 1):
        if inputNum % 2 == 0:
            inputNum = inputNum // 2
        else:
            inputNum = inputNum * 3 + 1
        count += 1

    return count

def unique_vals(inputDict):
    """
    Returns a list of unique elements from the
    input dictionary.
    """

    return list(set(inputDict.values()))
