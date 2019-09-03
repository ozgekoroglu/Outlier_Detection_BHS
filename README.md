# User Manual for Outlier Detection in Event Logs of Material Handling System
Source code for the graduation project "Outlier Detection in event logs of material handling system"

More information is available in the User Manual 

###Requirements

-- Install  Python 3 \
-- Install Anaconda \
-- Install  Python 

### Create a new environment

After Anaconda is installed, we can open the terminal Anaconda Prompt. If you'd like to use the graphical user interface, execute anaconda-navigator. Using the terminal Anaconda Prompt is quite straightforward. We can create virtual environments and install packages on them. It is convenient to create a specific virtual environment for a concrete project, to isolate the project and avoid affecting other programs with your modifications in the environment.

We start by creating a new environment. The command is :

```
conda create -n env_name
```

###	Activate the new environment

The new environment has been created in the directory we were located. To activate the new environment, type 

```
activate outlier_detection
```

###	Install packages

Once we have created an environment and it is active, we can install all the packages needed for our software to run.

To install all the required packages at once, since it can take some time, we provide the file requirements.txt with the list of packages. Then executing the following command will install all the packages one after the other:

•	For Windows:

```
FOR /F "delims=~" %f in (requirements.txt) DO conda install --yes "%f" || pip install "%f"
```
Hint: this command must be run from the folder where the requirements.txt file is located. Move to the given directory first (e.g. cd Desktop) or just add the path in the command, e.g. FOR /F "delims=~" %f in (Desktop/requirements.txt) DO conda install --yes "%f" || pip install "%f".

## EXECUTE THE OUTLIER DETECTION ALGORITHM

1)	Unzip the *Outlier_Detection_BHS-master* zip file.
2)	Unzip the *dist_clusters.rar* file
3)  (Optional) Execute the to obtain all the clusters that consists of similar days for each segment and weekday. These clusters are representative of the segment behavior. The result for each segment and weekday is stored in the folder called dist_cluster. We provide this folder with the results obtained for the period between 29.09.2017 and 30.03.2018 for Heathrow T3. Hence, execution of this step is not necessary unless someone wants to change the baseline for the analysis. If the baseline must be changed than the steps 1, 2, and 3 must be performed again. Execution is performed with following commands in the Anaconda prompt. 
```
(outlier_detection) >cd location_of_clustering.py
```
```
(outlier_detection) >python clustering.py
```
4)  Execute the *Outlier_Detection_in_BHS.py* with the argument that specifies the location of the segments we obtained in the step 2 with following commands in the Anaconda prompt. As a result of this step we obtain *test_day_weekday.csv* and *weekday.csv*  in *analysis_psm_general* and *blockage_psm_general folders*.
```
(outlier_detection) >cd location_of_Outlier_Detection_in_BHS.py
```
```
(outlier_detection) >python Outlier_Detection_in_BHS.py location_of_segments
```
5) Open the folder *psm_outlier*. Put the original event log that had been produced in the step 2 into this folder. Also put the *test_day_weekday.csv* file by changing its name to *outliers.txt*.

6) To prepare CSV file(s) for import, put the file(s) into a directory and provide a description as a text ini file with extension *.csvdir*. 

7) Execute *start.cmd* to open the Performance Spectrum Miner.

8) Import the .csvdir file via the **Open...** button.

9)	Choose parameters for generating the performance spectrum data. 

    •	The transformed data will be stored on disk in the Intermediate storage directory together with a meta-data file (session.psm).   You can load this transformed data also later via the Open... button.

    •	Type the following into the Custom Classifier Section. This classifier separates the outlier types in the performance spectrum miner.
  
    **org.processmininginlogistics.classifiers.bp.example.SegmentClassifierExample**
    
     • Choose **Process & open**
     •	The transformation may require some time and main memory depending on the Bin size chosen. Transformation for larger bin sizes are faster and require less memory.
     
10) Click open in the open pre-process dataset window.

11) Close the PSM.

12)	Open the *Intermediate storage directory*  for the performance spectrum that was obtained in the step 9. The intermediate storage directory is a path to an empty or non-existing folder where the performance spectrum data of the imported event data is stored. (Refer to user manual for the Performance Spectrum Miner for further information)

13) Add a *config.ini* file with the following content: \
__[GENERAL] \
   paletteId = 4__

14) Copy the *sorting_order.txt* file from the repository and paste it in the Intermediate storage directory.

15) Go to the **psm_outlier** folder and execute start.cmd again. But, this time select the *session.psm* in the Intermediate storage directory. 

16)	Explore the performance spectrum miner. For further information refer to the user manual for the PSM . Legend button shows the colors for different outlier types.

17) To see the problematic segments click on the Legend button and close it. A separate window is opened after closing the Legend window. To see the segments that behaved in their worst behavior select 1 for both comboboxes in the red rectangular and sort the values according to the importance variable.

18)	By clicking on any line, one can see the blockages regarding the segment for the given day in the blockage window.

19)	By clicking on any of the blockages, the user can see the selected blockage on the performance spectrum miner.

