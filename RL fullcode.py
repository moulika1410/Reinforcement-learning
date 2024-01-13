import random
import re
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from rake_nltk import Rake
import spacy
import string
from word2number import w2n
import xml.etree.ElementTree as ET
import tkinter as tk
from tkinter import filedialog
import pandas as pd




# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")


repo = {
    "alphabetic": "[[A-Z][a-z]]",
    "alphanumeric": "[[a-zA-Z][0-9]+]",
    "numeric": "[0-9]",
    "specialcharacters": "[[!#$%&'()+,-./:;<=>?@[]^_`{|}~]+]",
    "should contain": "(include)",
    "contain": "(include)",
    "contains": "(include)",
    "be": "(include)",
    "should not": "(exculde)",
    "must not": "(exclude)",
    "must be": "(include)",
    "must have": "(include)",
    "can be": "(include)",
    "cannot be": "(exclude)",
    "cannot" :"(exclude)",
    "in": "(include)",
    "and": "(include)",
    "only": "include",
    "or": "optional include",
    "not": "(exclude)",
    "if": "condition",
    "else": "condition",
    "any": "(include)",
    "minimum": "min length",
    "minimum of": "min length",
    "maximum": "max length",
    "length": "len",
    "greater than": ">",
    "less than": "<",
    "equal": "=",
    "greater than or equal to": ">=",
    "less than or equal to": "<=",
    "hifen": "-",
    "@": "[[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}]",
    "required": "(mandatory)",
    "optional": "exclude(not mandatory)",
    "regex": "(regular expression)",
    "day_pattern" : "[(mon|tue|wed|thu|fri|sat|sun)]",
    "date": "[(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/\d{4}]",
    "time": "[([01]\d|2[0-3]):[0-5]\d:[0-5]\d$]",
    "datetime": "[(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4} (0[1-9]|1[0-2]):[0-5]\d [APap][mM]]",
    "timezone": "[(GMT|[ECMP][DS]T|(?:[A-Z]+\/[A-Z_]+))]",
    "URL": "(value represents a valid URL)",
    "file format": "(required or allowed file format)",
    "uppercase": "[[A-Z]+]",
    "lowercase": "[[a-z]+]",
    "temperature":"[-?\d+(\.\d+)?[°]?[CFcf]]"
}


# Create the main tkinter window (root)
root = tk.Tk()
root.withdraw()  # Hide the main tkinter window

# Create a small separate popup window (top)
top = tk.Toplevel(root)
top.title("File Selection")

# Open a file dialog in the popup window
file_path = filedialog.askopenfilename(parent=top, filetypes=[("Text files", "*.txt"), ("All files", "*.*")])


# You can now access the selected file path using the 'file_path' variable
if file_path:
    print("Selected File Path:", file_path)

# Close the popup window
top.destroy()

# Prompt the user to enter a number between 1 and 10
user_input = input("Enter a number between 1 and 500: ")

# Validate the user input
number = None
try:
    number = int(user_input)
    if number < 1 or number > 500:
        raise ValueError
except ValueError:
    print("Invalid input. Please enter a number between 1 and 500.")
    exit()

#file_path = "C:\\Users\\ariselab\\Desktop\\US3.txt"
accepted_criteria = []

with open(file_path, "r") as file:
    lines = file.readlines()
    num_lines = len(lines)
    current_number = None
    current_criteria = ""
    for i in range(num_lines):
        line = lines[i].strip()
        if line.startswith(str(number) + "."):
            current_number = int(line.split(".")[0])
        elif current_number is not None:
            if line and not line.startswith(str(current_number + 1) + "."):
                current_criteria += line + "\n"
            else:
                if "Acceptance criteria:".lower() not in current_criteria.lower():
                    accepted_criteria.append(current_criteria.strip())
                current_criteria = ""
        if current_number is not None and line.startswith(str(current_number + 1) + "."):
            break

if not accepted_criteria:
    print("No acceptance criteria found for the given number.")
    exit()
# Print the acceptance criteria
for criteria in accepted_criteria:
    print(criteria)

STRING_LIST =accepted_criteria
# Extract keywords from the user story
keywords = set()
for text in STRING_LIST:
    # Split the text into words
    words = re.findall(r'\w+', text)
    # Add the words to the keywords set
    keywords.update(words)

# Update the repo dictionary with the extracted keywords
for keyword in keywords:
    if keyword not in repo:
        repo[keyword] = f"({keyword})"

# Print the updated repo dictionary
print("Updated repo:", repo)
print()

# Your code for loading stopwords remains the same
stopwords_file = r"C:\Users\Ariselab\Downloads\stopwords.txt"  # Provide the path to your stopwords file

with open(stopwords_file, "r") as file:
    stopwords_list = set(file.read().splitlines())

# Create a Rake object
r = Rake(stopwords=stopwords_list)

keywords_list = []

for text in STRING_LIST:
    for line in text.split('\n'):
        # Extract keywords from line
        r.extract_keywords_from_text(line)
        # Get the top 10 keywords
        keywords = r.get_ranked_phrases()

        # Check if there are keywords in the list
        if keywords:
            keywords_list.append(keywords)

# Print the keyword extraction
print("Keyword extraction:", keywords_list)
print()

# Calculate weights
weights = []
for keywords in keywords_list:
    q_values = [random.uniform(0, 1) for _ in keywords]  # Replace with your Q-value calculations
    q_sum = sum(q_values)
    weights.append([q_value / q_sum for q_value in q_values])

print("Weights:", weights)
print()

# Calculate rewards based on weights and constraints
rewards = []
for i in range(len(keywords_list)):
    reward = 0
    for j in range(len(keywords_list[i])):
        keyword = keywords_list[i][j]
        for repo_key in repo:
            if re.search(repo_key, keyword, re.IGNORECASE):
                # Adjust the reward based on the index of the repo key
                reward += weights[i][j] * (list(repo.keys()).index(repo_key) + 1)
                break
    rewards.append(reward)

# Normalize rewards to sum up to 100
total_reward = sum(rewards)
normalized_rewards = [reward * 100 / total_reward if total_reward != 0 else 0 for reward in rewards]

# Adjust the rewards based on weights
weighted_rewards = [normalized_rewards[i] * sum(weights[i]) for i in range(len(normalized_rewards))]
total_weighted_reward = sum(weighted_rewards)

if total_weighted_reward != 0:
    normalized_weighted_rewards = [reward * 100 / total_weighted_reward for reward in weighted_rewards]
else:
    normalized_weighted_rewards = [0] * len(weighted_rewards)

print("Normalized Rewards:", normalized_weighted_rewards)
print()


# Define a function to check if a string is a regular expression
def is_regex(string):
    return string.startswith("[") and string.endswith("]")

def generate_random_string(pattern, numeric_values):
    numeric_length = int(numeric_values)  # Convert numeric_values to an integer
    if pattern == "[[a-zA-Z][0-9]+]":
        return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(numeric_length))
    elif pattern == "[[A-Z][a-z]]":
        return ''.join(random.choice(string.ascii_letters) for _ in range(numeric_length))
    elif pattern == "[[A-Z]+]":
        return ''.join(random.choice(string.ascii_uppercase) for _ in range(numeric_length))
    elif pattern == "[[a-z]+]":
        return ''.join(random.choice(string.ascii_lowercase) for _ in range(numeric_length))
    elif pattern == "[0-9]":
        numeric_length = int(numeric_values) if numeric_values else 1  # Convert numeric_values to an integer or use a default length of 1
        return ''.join(random.choice(string.digits) for _ in range(numeric_length))
    elif pattern == "[(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/\d{4}]":
        day = str(random.randint(1, 31)).zfill(2)
        month = str(random.randint(1, 12)).zfill(2)
        year = str(random.randint(1950, 2023))
        return f"{month}/{day}/{year}"
    elif pattern == "[(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4} (0[1-9]|1[0-2]):[0-5]\d [APap][mM]]":
        day = str(random.randint(1, 31)).zfill(2)
        month = str(random.randint(1, 12)).zfill(2)
        year = str(random.randint(1950, 2023))
        hour = str(random.randint(1, 12)).zfill(2)  # 12-hour format
        minute = str(random.randint(0, 59)).zfill(2)
        am_pm = random.choice(['AM', 'PM'])
        return f"{day}/{month}/{year} {hour}:{minute} {am_pm}"
    elif pattern == "[([01]\d|2[0-3]):[0-5]\d:[0-5]\d$]":
        hour = str(random.randint(0, 23)).zfill(2)
        minute = str(random.randint(0, 59)).zfill(2)
        second = str(random.randint(0, 59)).zfill(2)
        return f"{hour}:{minute}:{second}"
    elif pattern == "[(mon|tue|wed|thu|fri|sat|sun)]":
         options = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]  # List of valid day options
         return random.choice(options)
    elif pattern == "[(GMT|[ECMP][DS]T|(?:[A-Z]+\/[A-Z_]+))]":
        options = ["GMT", "EST", "EDT", "CST", "CDT"]  # Add more timezone options as needed
        return random.choice(options)
    elif pattern == "[[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}]":
        #username = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        domain = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
        return f"@{domain}.com"
    elif pattern == "[[!#$%&'()+,-./:;<=>?@[]^_`{|}~]+]":
        return ''.join(random.choice("!#$%&'()+, -./:;<=>?@[]^_`{|}~") for _ in range(numeric_length))
    elif pattern == "[-?\d+(\.\d+)?[°]?[CFcf]]":
        temperature = random.uniform(-100.0, 100.0)  # You can adjust the range as needed
        # Randomly choose either Celsius (C) or Fahrenheit (F)
        unit = random.choice(["C", "F"])
        # Format the temperature string
        return f"{temperature:.2f}{unit}"
    else:
        # Handle unrecognized patterns here (return a default value or raise an exception)
        return None

def generate_random_invalid(pattern, numeric_values):
    numeric_length = int(numeric_values)  # Convert numeric_values to an integer
    if pattern == "[[a-zA-Z][0-9]+]":
        return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(numeric_length - 1))
    elif pattern == "[[A-Z][a-z]]":
        return ''.join(random.choice(string.ascii_letters) for _ in range(numeric_length - 1))
    elif pattern == "[[A-Z]+]":
        return ''.join(random.choice(string.ascii_lowercase) for _ in range(numeric_length - 1))
    elif pattern == "[[a-z]+]":
        return ''.join(random.choice(string.ascii_uppercase) for _ in range(numeric_length - 1))
    elif pattern == "[0-9]":
        numeric_length = int(numeric_values) if numeric_values else 1  # Convert numeric_values to an integer or use a default length of 1
        return ''.join(random.choice(string.digits) for _ in range(numeric_length - 1))
    elif pattern == "[(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/\d{4}]":
        day = str(random.randint(30, 40)).zfill(2)
        month = str(random.randint(12, 20)).zfill(2)
        year = str(random.randint(1950, 2023))
        return f"{month}/{day}/{year}"
    elif pattern == "[[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}]":
        #username = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        domain = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
        return f"@{domain}.com"
    elif pattern == "[(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4} (0[1-9]|1[0-2]):[0-5]\d [APap][mM]]":
        day = str(random.randint(30, 40)).zfill(2)
        month = str(random.randint(1, 12)).zfill(2)
        year = str(random.randint(1950, 2023))
        hour = str(random.randint(1, 12)).zfill(2)  # 12-hour format
        minute = str(random.randint(0, 59)).zfill(2)
        am_pm = random.choice(['AM', 'PM'])
        return f"{day}/{month}/{year} {hour}:{minute} {am_pm}"
    elif pattern == "[([01]\d|2[0-3]):[0-5]\d:[0-5]\d$]":
        hour = str(random.randint(23, 24)).zfill(2)
        minute = str(random.randint(59, 69)).zfill(2)
        second = str(random.randint(0, 59)).zfill(2)
        return f"{hour}:{minute}:{second}"
    elif pattern == "[[!#$%&'()+,-./:;<=>?@[]^_`{|}~]+]":
        return ''.join(random.choice("!#$%&'()+, -./:;<=>?@[]^_`{|}~") for _ in range(numeric_length -1))
    else:
        # Handle unrecognized patterns here (return a default value or raise an exception)
        return None

# Function to process keywords based on "exclude" and "include" and print numeric representations
def process_keywords(keywords_list):
    result = [] 
    exclude_mode = False

    for keyword in keywords_list:
        if keyword.lower() == '(exclude)':
            exclude_mode = True
        elif keyword.lower() == '(include)':
            exclude_mode = False
        elif not exclude_mode:
            # Check if the keyword contains a word inside '( )'
            match = re.search(r'\((\w+)\)', keyword)
            if match:
                word_inside_parentheses = match.group(1)
                try:
                    numeric_representation = w2n.word_to_num(word_inside_parentheses)
                    keyword = keyword.replace(match.group(), f'({numeric_representation})')
                except ValueError:
                    # Handle cases where word_to_num cannot convert
                    pass
            result.append(keyword)

    return result


# Initialize lists to store valid and invalid test cases
valid_testcases = []
invalid_testcases = []

# Loop through each accepted criteria
for S in STRING_LIST:
    # Split the line into words using regex to handle punctuation
    lines = S.split('.')
    
    for line in lines:
        # Split the line into words
        words = re.findall(r'\b\w+\b|[^\w\s]', line.lower())
        
        # Check if there are words in the line
        if words:
            # Process each word
            for word in words:
                # If the word is in the repo, replace it with its meaning
                if word in repo:
                    line = line.replace(word, repo[word])

            # Print the updated line
            print("List split:", words)
            
        extraction =[]
        for word in words:
            if word in repo:
                extraction.append(repo[word])
        
        # Check if there are any keywords in the extraction list
        if not extraction:
            continue
        print("Keywords matches repo:", extraction)
        # Process the extraction for "exclude" and "include" keywords
        include_keywords = process_keywords(extraction)
        print("Included keywords after processing 'exclude' and 'include':", include_keywords)
        regex_keywords = [keyword for keyword in include_keywords if is_regex(keyword)]
        
        
        # Create a list to store the valid and invalid random strings for this line
        valid_random_strings = []
        invalid_random_strings = []

        if regex_keywords:
            # Extract numeric values enclosed in parentheses
            numeric_values = [match.group(1) for keyword in include_keywords if (match := re.search(r'\((\d+)\)', keyword))]
            for regex_keyword, numeric_value in zip(regex_keywords, numeric_values):
                print("Regular expression:", regex_keyword)
                print("Numeric Value:", numeric_value)
                
                # Generate a random string that matches the regular expression
                valid_random_string = generate_random_string(regex_keyword, numeric_value)
                invalid_random_string = generate_random_invalid(regex_keyword, numeric_value)
                if valid_random_string is not None:
                    valid_random_strings.append(valid_random_string)
                    print("Valid:", valid_random_string)
                    
                if invalid_random_string is not None:
                    invalid_random_strings.append(invalid_random_string)
                    print("Invalid:", invalid_random_string)

            print()
            
            
            # Combine the valid random strings for this line into one string
            valid_random_string_combined = ''.join(valid_random_strings)
            invalid_random_string_combined = ''.join(invalid_random_strings)
            
            # Append the combined valid and invalid random strings to the respective lists
            valid_testcases.append(valid_random_string_combined)
            invalid_testcases.append(invalid_random_string_combined)
        else:
             print("No regex keywords found for this line. So please rewrite the acceptance criteria with certain conditions")
             print()
            
# Print all valid test cases
print("Valid test cases:")
for i, testcase in enumerate(valid_testcases, start=1):
    print(f"Test case {i}: {testcase}")

# Print all invalid test cases
print("\nInvalid test cases:")
for i, testcase in enumerate(invalid_testcases, start=1):
    print(f"Test case {i}: {testcase}")



# Create an XML element for valid test cases
valid_testcases_element = ET.Element("ValidTestCases")

# Add valid test cases as child elements
for i, testcase in enumerate(valid_testcases, start=1):
    testcase_element = ET.SubElement(valid_testcases_element, f"TestCase_{i}")
    testcase_element.text = testcase

# Create an XML element for invalid test cases
invalid_testcases_element = ET.Element("InvalidTestCases")

# Add invalid test cases as child elements
for i, testcase in enumerate(invalid_testcases, start=1):
    testcase_element = ET.SubElement(invalid_testcases_element, f"TestCase_{i}")
    testcase_element.text = testcase

# Create the root XML element
root_element = ET.Element("TestCases")

# Add valid and invalid test cases elements as children of the root
root_element.append(valid_testcases_element)
root_element.append(invalid_testcases_element)

# Create an XML tree from the root element
tree = ET.ElementTree(root_element)

# Save the XML tree to a file
xml_filename = "D:\\Deep learning\\testcases.xml"
tree.write(xml_filename)

print(f"Valid and invalid test cases have been saved to '{xml_filename}' XML file.")

# Create a text file for writing the test cases
text_filename = "textfile.txt"

# Write valid test cases to the text file
with open(text_filename, "w") as text_file:
    text_file.write("Valid Test Cases:\n")
    for i, testcase in enumerate(valid_testcases, start=1):
        text_file.write(f"TestCase_{i}: {testcase}\n")

    # Add a newline between valid and invalid test cases
    text_file.write("\n")

    # Write invalid test cases to the text file
    text_file.write("Invalid Test Cases:\n")
    for i, testcase in enumerate(invalid_testcases, start=1):
        text_file.write(f"TestCase_{i}: {testcase}\n")

print(f"Valid and invalid test cases have been saved to '{text_filename}' text file.")
# Create a new DataFrame for storing each pair in a separate row
output_df = pd.DataFrame(columns=['Test Case ID', 'Test Case Name', 'Component', 'Priority', 'Summary', 'Pre-Requisite', 'Steps', 'Step Actions', 'Expected Result'])

# Open the text file for writing valid and invalid test cases
with open(text_filename, "w") as text_file:
    text_file.write("Test Cases:\n")

    for i, (valid_testcase, invalid_testcase) in enumerate(zip(valid_testcases, invalid_testcases), start=1):
        # Valid Test Cases
        valid_testcase_element = ET.SubElement(valid_testcases_element, f"TestCase_{i}")
        valid_testcase_element.text = valid_testcase

        output_df = output_df.append({'Test Case ID': user_input, 'Test Case Name': accepted_criteria, 'Component': '', 'Priority': '', 'Summary': "",
                                      'Pre-Requisite': "", 'Steps': f"{valid_testcase}", 'Step Actions': 'Valid or Invalid', 'Expected Result': 'Valid'}, ignore_index=True)

        # Write both valid and invalid test cases to the text file
        text_file.write(f"TestCase_{i} (Valid): {valid_testcase}\n")
        text_file.write(f"TestCase_{i} (Invalid): {invalid_testcase}\n")

        # Invalid Test Cases
        output_df = output_df.append({'Test Case ID': user_input, 'Test Case Name': accepted_criteria, 'Component': '', 'Priority': '', 'Summary': "",
                                      'Pre-Requisite': "", 'Steps': f"{invalid_testcase}", 'Step Actions': 'Valid or Invalid', 'Expected Result': 'Invalid'}, ignore_index=True)

# Write to Excel file with a single column for both valid and invalid values
output_file_path = r"C:\Users\Ariselab\Downloads\testcases.xlsx"
with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
    output_df.to_excel(writer, sheet_name='Sheet1', index=False, startcol=0, startrow=0, columns=['Test Case ID'])
    output_df.to_excel(writer, sheet_name='Sheet1', index=False, startcol=1, startrow=0, columns=['Test Case Name'])
    output_df.to_excel(writer, sheet_name='Sheet1', index=False, startcol=2, startrow=0, columns=['Component'])
    output_df.to_excel(writer, sheet_name='Sheet1', index=False, startcol=3, startrow=0, columns=['Priority'])
    output_df.to_excel(writer, sheet_name='Sheet1', index=False, startcol=4, startrow=0, columns=['Summary'])
    output_df.to_excel(writer, sheet_name='Sheet1', index=False, startcol=5, startrow=0, columns=['Pre-Requisite'])
    output_df.to_excel(writer, sheet_name='Sheet1', index=False, startcol=6, startrow=0, columns=['Steps'])
    output_df.to_excel(writer, sheet_name='Sheet1', index=False, startcol=7, startrow=0, columns=['Step Actions'])
    output_df.to_excel(writer, sheet_name='Sheet1', index=False, startcol=8, startrow=0, columns=['Expected Result'])

print(f"Valid and invalid test cases have been saved to '{text_filename}' text file and '{output_file_path}' Excel file.")
