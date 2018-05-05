###################################################################
# wav2mfcc.praat        revised March 13, 2013
# Jeff Mielke
# read all the wav files in a directory, make mfcc objects from them, and save as matrices
###################################################################

# CHANGE THIS TO YOUR DIRECTORY
directory$ = "/home/luisrobles/Dev/Thesis/data/SAVEE"
out_directory$ = "/home/luisrobles/Dev/Thesis/data/SAVEEmfcc"

#form Variables
#	sentence directory
#	sentence out_directory
#endform

# FIND ALL WAV FILES IN THE DIRECTORY
Create Strings as file list... list 'directory$'/*.wav
numberOfFiles = Get number of strings

# LOOP THROUGH ALL THE WAV FILES
for ifile to numberOfFiles

    #OPEN THEM
    select Strings list
    filename$ = Get string... ifile
    baseFile$ = filename$ - ".wav"
    Read from file... 'directory$'/'baseFile$'.wav

    #CONVERT THEM AND SAVE THEM
    select Sound 'baseFile$'
    #Pre-emphasize (in-line)... 50
    To MelFilter... 0.015 0.005 100 100 0
    To MFCC... 12
    To TableOfReal... yes
    #To Matrix

    Write to text file... 'out_directory$'/'baseFile$'.txt
    #appendInfoLine(TableOfReal)

    #CLEAN UP
    select MelFilter 'baseFile$'
    plus MFCC 'baseFile$'
    #plus Matrix 'baseFile$'
    plus TableOfReal 'baseFile$'
    plus Sound 'baseFile$'
    Remove

endfor

select Strings list
Remove
