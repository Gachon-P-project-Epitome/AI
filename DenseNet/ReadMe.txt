
1) Preparing Data
    Step 1
The FMA dataset, fma_smal.zip, can be downloaded from https://github.com/mdeff/fma.
The path to a decompressed fma_small.zip is "./data/fma_small/," and after decompressing, the data is *.mp3 files in different directories.
Convert *.mp3 files to wave files using proc_fma.py, utils.py, and tracks.csv. Before, the tool for converting MP3 to wave files
must be installed, Download this tool, FFmpeg, from https://ffmpeg.org/download.html. After Step 1, all wave files are located in the
directory Samples with eight subdirectories using the names of eight genres. The sampling frequency of these wave files is 44100 Hz with 32 bits per sample. 
(float format)
    
    Step 2
Use ConvertWav32FL_2_16PCM.py to lower the sampling frequency to 16000 Hz and 16 bits/sample (int16 format). The directory for downsampling frequency files
is FMA_16B
    
    Step 3
Create echo wave files using FMA_Create_Echo.py. These echo wave files should be located in the directory /FMA_16B/ECHO.
    
    Step 4
Create image files (*.png) for 8000 original wave files and for data augmentation. After finishing this step,4XDATA = 4X8000 = 32000 image files are located 
in the same directory, "FMA_IMAGES," used for training and testing.
Using Convert_FMA_16kHz_2_Images_ORG_ORGNOISE.py, create image files for original files and original files with noise.
Convert_FMA_16kHz_2_Images_ECHO_ECHONOISE.py is used to create image files for echoed files and echoed files with noise.

2) Training and Testing
Training and testing are done with DenseNet121, DenseNet169, and DenseNet201 using (Training_FMA_DenseNet121.py and Testing_FMA_DenseNet121.py), 
(Training_FMA_DenseNet169.py, Testing_FMA_DenseNet169.py), and (Training_FMA_DenseNet201.py, Testing_FMA_DenseNet201.py), respectively.


---------------------
If you encounter any problems, send an email to loan.trinhvan@hust.edu.vn or thuydtl@utc.edu.vn 