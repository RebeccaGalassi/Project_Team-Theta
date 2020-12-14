# Project_Team-Theta

By team Theta
Introduction 
This project aims at producing an algorithm capable of predicting the outcome of a premier league game and giving advice for betting on a match and match week selected by the user. In order to achieve that result, we took data for the period 2000-2015. The data analyzed features the results of every premier league game played during the interval. Data for the matchweek is shown to the user based on the matchweek they desire.

PROJECT STRUCTURE
In the final repository you will find the following elements:
-DataCleaning.ipynb
-Main.ipynb
-Final.py
-Football_prediction.py
-Data Folder
-Test Folder

The first two files (DataCleaning.ipynb and Main.ipynb) are two jupyter notebooks which were used during the development of the various functions but are not necessary for the final application. They’ve been left in the final repository to let the user understand the various components. 

DataCleaning.ipynb is probably the densest file of the project. Its job is to clean the data for prediction. To condense and prepare the data for prediction, we first had to eliminate irrelevant data from the datasets. After skimming by dropping columns the data had to be organized, divided into manageable intervals, that is, match weeks. To help the prediction further, we gave better significance to the data by adding the cumulative goals scored and goals conceded by each team during the course of one season. 

Main.ipynb is the product of the data cleaning where we perform the prediction.
Final.py  is the part the final user interacts with. When opened, the user is asked which matchweek they would like to visualize. It then provides a list of the matches in the desired week, along with the outcome the program thinks is most likely to win, therefore the most favourable to bet on.

The data folder contains all the csv files used for the application and must be saved on the desktop with the path below (data folder must not be contained in other folders).
'C:/Users/Utente/Desktop/Data
The sources of the files are the followings:
 http://football-data.co.uk/data.php
English premier league tables | Kaggle
In addition it contains a text file where are listed the meanings of the most important features.
Test folder contains a file called test.py that tests the input of the prediction function of the Football_prediction.py file.

The test folder contains the files __init__.py and test_final.py and the aim is to test the application input on the prediction function.

Football_prediction.py contains the prediction algorithms used. We utilized an array of functions in order to better gauge their effectiveness. We found the most effective one was the xgboost method. If you are interested in better understanding how the mathematical model works this link will lead you to the official documentation that contains also additional information on the package and how to install it in various systems.

XGBoost Documentation — xgboost 1.4.0-SNAPSHOT documentation 

Xgboost is the only library not included in Python used in the project (see the how to section for install instructions)

HOW TO INSTALL XGBOOST
Step 1 
First, you need to download python 64bit  on your computer.
 
Get the latest Python version :https://www.python.org/downloads/windows/
 
If you already have Python installed, you need to check to have the right version: open command prompt and run following command to get the version of python installed on your system. (if you use Visual Studio Code and install the python extension, when you open the files on the bottom left a blue section with the version number should appear).

 python –version
 
Step 2 
You need to install pip, to get it visit:
https://bootstrap.pypa.io/get-pip.py
Save the file above and run this command:
python get-pip.py
 
Step 3
Now that you have pip installed get Python XGBoost using the link below
https://www.lfd.uci.edu/~gohlke/pythonlibs/#xgboost
Install the downloaded XGBoost file using the command below 
pip install “the name of the downloaded file “
Make sure to match your python version and system architecture, e.g. “xgboost-0.81-cp37-cp37m-win_amd64.whl” for python 3.7 on a 64-bitmachine.

In the example:
pip install xgboost-0.81-cp37-cp37m-win_amd64.whl
That’s all. You have successfully installed XGBoost.

HOW DOES IT WORK
When the user open the Final.py file runs it and must insert an integer number from 1 to 38 in addiction to other arguments:
-o to save the result into a csv file (optional )
-v or -q to have a full explanation or just the suggestion. Pay attention these are mutually exclusive parameters meaning that you have to choose between the two to insert. 

Contributors: 
Rebecca Galassi (prediction, argparse and csv )
Edoardo Tasca (PEP8)
Carlo cremona (readme and feature engineering )
Loriana Dudau (data-cleaning  and feature engineering )
Nicolo Mariconda (data-cleaning)
