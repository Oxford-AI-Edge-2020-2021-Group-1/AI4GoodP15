The main MATLAB 2020b live script named 'main_v2' will run in order to obtain audio snippets from the ELP audio files that may be gunshots. 
The other .m function files are called upon in the main_v2.
A folder with ELP is required in the MATLAB directory to be called on. 

The Audio Toolbox by MATLAB is required to run 'main_v2' properly. 

1. Problem Analysis / Data preview
In order to better collect training, validation, and testing audio files, it is best to collate the ELP audio files for snippets where gunshots are heard.
The challenge is the size of the ELP audio files (approximately 1.3Gb each). The majority of the audio file is of forest noise and animal sounds. 
Hence, this matlab script mains to collate audio snippets of interest to pass onto the Python training models.

2. How the script works
\n 2.1 It reads the folder where the audio files are stored. 
\n 2.2 It audioreads 1 hour of 1 folder into smaller segments of 2mins, resulting in a matrix where columns of minutes and rows is the audio at that timestep within the 2minutes.
\n 2.3 A spectrogram was created on the audio snippet.
\n 2.4 detectSpeech is used from the Audio Toolbox to detect regions of interest within the 2min snippet to be stored and used in the classification models. 
