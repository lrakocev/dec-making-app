# breakdown_stories.py README
-----------------------------------------------------------------------------------
WHAT THIS SCRIPT IS:

This script takes a Word document (with a '.docx' extension) and breaks it down
into several text files that comply with the format the decision-making (DM) app
accepts. For each story in the Word file, it'll produce the following files:
	
	- context.txt
	- pref_cost.txt
	- pref_reward.txt
	- questions.txt

The script will also create the same folder structure the DM app needs, so each
story will be contained in '/output/{task type}/story_{story #}/', where the
curly braces ( {} ) represent a placeholder. Each of the text files will be
contained in '/output/{task type}/story_{story #}/', just how the app needs them.

-----------------------------------------------------------------------------------
WHAT THIS SCRIPT IS NOT:

This script does not check for grammar, spelling, or punctuation. Whatever is in
the Word file you give it, it will take directly. It will only remove certain
pieces of text (for example the numbers on a list), but it will make no other
modifications to whatever is in the Word file. This means that if the Word document
is missing anything, it will be missing in the generated text files as well.

For this reason, make sure that your Word documents are complete before passing them
through this script, or you will get blank or incomplete text documents.

-----------------------------------------------------------------------------------
HOW TO USE THIS SCRIPT:

SETUP:

1. Use the template I've provided to write your stories.
2. Finish writing all stories, preferences, and questions throughout the whole
	document.
3. Follow this naming convention: '{Task_type} DM.docx'. The first few words in
	the file name should describe the task type. You do not need to capitalise
	the first letter, but do separate each word with an underscore (_). Follow
	the task type with a space and 'DM'. Save the Word file as a .docx ONLY.
4. Place this file in the '/input/' folder.

RUNNING THE SCRIPT:

5. Double click on the file 'startup.bat'. A black command window will pop up.
6. Sit back and relax while the script does its job!
	The script might take longer if you feed it many word files, but it'll do
	its best to finish all the work. You'll see what Word files it's working on
	and when it's done processing each one.
7. Open up the '/output/' folder to find all the stories separated into the files
	the app accepts, formatted exactly how it wants them to be. Look over the text
	files make sure they all look good.
	Take the contents of the '/output/' folder and transfer them to the DM app.
8. In the DM app folder, navigate to '/stories/task_types/'. Place the contents of
	'/output/' in here.
9. You're done! You just did several hours of work in just a few seconds!

# verify_files.py README
-----------------------------------------------------------------------------------
Once all stories are broken down, it is recommended to run them through verify_files.
This script will attempt to process the text files as the main app would, but it will
note any errors and problems it finds and will display them on the command line.

Running verify_output.bat will automatically verify the integrity of all files in the
'output' directory. Warnings and errors will be displayed, along with an indication of
where the error was found on the file.

One must *manually* fix the issues afterward.

RUNNING THE SCRIPT MANUALLY:

The script accepts four (4) arguments...
1. root: The root directory/folder where story data to be checked is stored. This can
	either be 'output' for the output folder, or 'stories' for the main stories
	folder.
2. -t, --task: Task type directory/folder to analyse. If set, the program will restrict
	itself to only the specified task type. Please only choose one of the following:
	['approach_avoid', 'benefit_benefit', 'cost_cost', 'moral', 'multi_choice', 'probability', 'social']
3. -s, --story: Story directory/folder to analyse. If set, the program will restrict
	itself to only the specified task type. Only enter an integer number.
4. -f, --file: File to restrict analysis to.
	Choices are ['questions', 'pref_cost', 'pref_reward', 'context']

Running the script indicating only the 'root' argument result in a verification of
*all* files in the indicated root folder. One can restrict this to a specific file
by providing the remaining three optional arguments. However, all three must be
provided.
