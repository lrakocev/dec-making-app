# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:19:52 2024

@author: Raquel
Verifies file integrity of stories.
"""

import re
from copy import deepcopy
from os import path, listdir
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from textwrap import dedent

def read_dir_map(rootdir, restrict_numeric=False, get_full_filenames=False):
# Returns the structure of the given directory as a dictionary.
    
    dir_map = {}
    # Get the task types directories as a list.
    # folders = listdir(rootdir)
    folders = [ name for name in listdir(rootdir) if path.isdir(path.join(rootdir, name)) and (name.isnumeric() if restrict_numeric else True) ]
    
    for folder in folders:
        # Get the directories inside each task type directory as a list.
        dirs = listdir(rootdir+'/'+folder)
        if len(dirs) > 0:
            # If the directory is populated...
            for i in range(len(dirs)):
                # Take only the story number from the directory names...
                dirs[i] = dirs[i] if get_full_filenames else dirs[i].split('_')[-1]
        # ...then populate the dictionary with 'task type': 'stories for
        # in that task type'.
        dirs.sort(reverse=True)
        dir_map.update({folder: dirs})

    return dir_map

def read_file(root_dir, task, story, file):
    # Open, read, and process text lines
    file_path = path.abspath(f"{root_dir}/{task}/{story}/{file}.txt")
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"\nVerifying integrity of file '{file_path}'...")
    lines = open(file_path).read().split("\n")
    lines = [line.strip() for line in lines]
    i = 0
    errors = []
    # Process each file differently:
    if file in ['pref_cost', 'pref_reward']:
        opt_num = 9999
        opt_description = ""
        opt_dict = {}
        for i, option in enumerate(lines):
            try:
                if option.strip() == '':
                    opt_dict[opt_num] = ""
                    raise Exception("blank line")
                opt_num_old = deepcopy(opt_num)
                opt_description_old = deepcopy(opt_description)
                line = option.split(")")
                opt_num = int(line[0])
                opt_description = line[1].strip()
                opt_dict[opt_num] = opt_description
                if opt_num == opt_num_old or opt_description == opt_description_old:
                    raise Exception("repeated item")
            except Exception as e:
                error_tup = (i, e, 'err')
                errors.append(error_tup)
        
        if i != 5 and i != 11:
            errors.append(("file",f"incorrect number of items: looking for 6 or 12, got {i+1}",'err'))
    
    elif file == 'context':
        try:
            ' '.join(lines)
        except Exception as e:
            errors.append('x', e, 'err')
        
    elif file == 'questions':
        # Pattern to match all sentences, regardless of punctuation (curly brace included for debugging with example stories).
        pattern = re.compile(r'([A-Z0-9][^\.!?}]*[\.!?}])', re.M)
        question = ""
        RC = ""
        quest_dict = {}
        for i, l in enumerate(lines):
            try:
                # Look for blank spaces
                if l.strip() == '':
                    quest_dict.update({ "blank": "blank"})
                    raise Exception("blank line")
                    
                # Save old entry to compare to new
                question_old = question
                RC_old = RC
                
                # Look for complete sentences; will fail if not punctuation is present or if first word is not capitalised
                linelist = ['', '']
                linelist[0] = ' '.join(pattern.findall(l)) # Find all sentences in the question
                
                # Try to get the cost and reward level tag
                linelist[1] = re.findall('\(.*?\)', l)[-1]  # Find everything in parenthesis and take the last element.
                question = linelist[0]
                RC = re.findall("\d+", linelist[1]) # Take only the numeric values in this part of the string.
                keytup = ()
                for x in RC:
                    keytup = keytup + (int(x),)
                quest_dict.update({ keytup: question })
                
                # Try to separate question into two for certain task types
                if task in ['multi_choice', 'benefit_benefit', 'cost_cost']:
                    match = re.search(r'\bor\b', l, flags=re.IGNORECASE)
                    if not match:
                        raise Exception("could not find delimiter word 'or' to split question into two equal parts")
                    split_question = question.split(' or ')
                    if len(split_question[0]) - len(split_question[-1]) > 118:
                        errors.append((i, f"length of question option A (n={str(len(split_question[0]))} characters) is significantly greater than length of option B (n={str(len(split_question[1]))} characters)", 'warn'))
                
                # Raise flag if new question matches the old one
                if question == question_old or RC == RC_old:
                    raise Exception("repeated item")
                    
            except Exception as e:
                error_tup = (i, e, 'err')
                errors.append(error_tup)

        if i != 35:
            errors.append(("file",f"incorrect number of items: looking for 36, got {i+1}", 'err'))
            
    if len(errors) > 0:
        print("  Errors were found when processing file:")
        for x, item in enumerate(errors):
            print(f"    {x+1}. {'[ERROR]:' if item[2] == 'err' else '[WARNING]:'} '{item[1]}' on {'line '+str(item[0]) if str(item[0]).isnumeric() else 'whole file'}.")
    else:
        print("  No errors were detected!")

if __name__ == "__main__":
    # ------------------------- Parse cmd arguments -------------------------------

    # Define the parser
    argparser = ArgumentParser(prog='HUMANSIntegrityChecker', formatter_class=RawDescriptionHelpFormatter,
                               description=dedent('''\
     ----------------------------------
    |  H.U.M.A.N.S. integrity checker  |
    |----------------------------------|
    |                                  |
    | Verifies the integrity of story  |
    | files and warns the user of      |
    | potential problems with the      |
    | files.                           |
    |                                  |
     ----------------------------------
     '''))
    
    # Declare an argument, using a default value if the argument 
    # isn't given
    argparser.add_argument('root', help="the root directory/folder where story data to be checked is stored", choices=['output', 'stories'])
    argparser.add_argument('-t', '--task', action="store", dest='task', help="task type directory/folder to analyse. If set, the program will restrict itself to only the specified task type.", default=None, choices=['approach_avoid', 'benefit_benefit', 'cost_cost', 'moral', 'multi_choice', 'probability', 'social'])
    argparser.add_argument('-s', '--story', action="store", dest='story', help='story directory/folder to analyse. If set, the program will restrict itself to only the specified task type.', default=None)
    argparser.add_argument('-f', '--file', action="store", dest="file", help='file to restrict analysis to', choices=['questions', 'pref_cost', 'pref_reward', 'context'], default=None)
    # Now, parse the command line arguments and store the 
    # values in the 'args' variable
    args = argparser.parse_args()

    # ------------------------------ Setup ----------------------------------------
    
    print("\n>>> Script started, please wait... <<<")
    root = path.abspath('../../stories/task_types' if args.root=='stories' else 'output')
    filepath = path.abspath(f"{root}{'/'+str(args.task) if not args.task is None else ''}")
    dirlist = read_dir_map(rootdir=filepath, restrict_numeric=False, get_full_filenames=True)
    if args.task is None:
        files = ['questions', 'pref_cost', 'pref_reward', 'context']
        for folder, subfolder in dirlist.items():
            for story in subfolder:
                for file in files:
                    if folder == 'benefit_benefit' and file == 'pref_cost':
                        pass
                    elif folder == 'cost_cost' and file == 'pref_reward':
                        pass
                    else:
                        read_file(root, folder, story, file)
    else:
        read_file(root, args.task, f"story_{str(args.story)}", args.file.split('.')[0])
    
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("\n>>> Verification done! Please review the log and manually correct issues detected. <<<\n\n")