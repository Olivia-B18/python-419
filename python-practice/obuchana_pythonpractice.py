'''
Olivia Buchanan
Python Practice
'''

#####################################
# SET VARIABLES
#####################################

HAL = 9000
Bridge = "4"
Nahn = [2]
R2 = ["D", "2"]
C = [{"3PO"}]
Poppins = set("supercalifragilisticexpialidocious")
WALL = [["E"]]
K = {"2SO"}

#####################################
# SET DICTIONARIES
#####################################

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

#####################################
# SET UNIVERSITY LEVELS
#####################################

Germanna_levels = ["Freshman", "Sophomore"]
UMW_levels = ["Freshman", "Sophomore", "Junior", "Senior"]
MaryWash_levels = UMW_levels
CNU_levels = UMW_levels.copy()

#######################################
# FUNCTIONS
########################################
def plus2(num: int):
    """
    Return input num plus 2.

    Parameters:
        num (int): Input number.

    Returns:
        int: Sum of num and 2.
    """
    return num + 2

DAT_SET = {"Navani", "Shallan", "Jasnah"}
def gimme_dat_set():
    """
    Return the same set object.

    Parameters: None

    Returns:
        set: Constant DAT_SET
    """
    return DAT_SET

def gimme_set_like_dat():
    """
    Return a different set object but with same values every call.

    Parameters: None

    Returns:
        set: A new set with the same values.
    """
    return {"Kaladin", "Dalinar", "Adolin"}

def center(input_string: str):
    """
    Return middle letter(s) of a string

    Parameters:
        input_string (str): Input string.

    Returns:
        str: The middle letter(s) of input_string
    """
    strlen = len(input_string)
    if strlen == 0: # Empty string
        return None
    elif strlen % 2 == 1: # Odd length
        return input_string[strlen // 2]
    else: # Even length
        return input_string[strlen // 2 - 1:
                           strlen // 2 + 1]

def nuke_last(input_list: list, input_element: str):
    """
    Remove last instance of an element in a list.

    Parameters:
        input_list (list): Input list.
        input_element (str): Element to remove.

    Returns: None
    """
    if input_element in input_list:
        input_list.reverse()
        input_list.remove(input_element)
        input_list.reverse()

def middlest(num1: int, num2: int, num3: int):
    """
    Returns the middle number of three numbers, in order.

    Parameters:
        num1 (int): First number.
        num2 (int): Second number.
        num3 (int): Third number.

    Returns:
        int: the middle number of the three numbers, in order.
    """
    nums = [num1, num2, num3]
    nums.sort()
    return nums[1]

def tack_on_end(input_list: list,
                add_this: list | str | int | float,
                times_to_repeat: int = 1):
    """
    Append the item(s) in add_this to list1.

    Parameters:
        input_list (list): Input list.
        add_this (list | str | int | float): Item(s) to append.
        times_to_repeat (int): Number of times to append.

    Returns: None
    """
    for _ in range(times_to_repeat):
        if isinstance(add_this, list):
            input_list.extend(add_this)
        else:
            input_list.append(add_this)

def wondrous_count(num: int):
    """
    Give the number of steps needed to reach 1 using Collatz Conjecture.

    Parameters:
        num (int): The starting number.

    Returns:
        int: The number of steps to reach 1 using Collatz Conjecture.
    """
    steps = 0
    while num > 1:
        if num % 2 == 0:
            num = num // 2
        elif num % 2 == 1:
            num = num * 3 + 1
        steps += 1
    return steps

def unique_vals(input_dict: dict):
    """
    Return a list of the unique values in input_dict.

    Parameters:
        input_dict (dict): Input dictionary.

    Returns:
        list: unique values in input_dict
    """
    return list(set(input_dict.values()))

