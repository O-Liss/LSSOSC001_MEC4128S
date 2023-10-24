# LSSOSC001_MEC4128S
 FINAL YEAR PROJECT - SEA ICE ANALYSIS

Abstract
The annual growth and retreat of sea ice in Antarctica is one of the one most expansive seasonal events on the planet. ASPeCt conduct ship-based analyses, using human observations to understand and document the physical, chemical and biological processes taking place in the marginal ice zone of Antarctica. While the South African Weather Service are required to document safe shipping routes defined by the edge of the Marginal ice zone for the international shipping community. 

The human observations carried out by ASPeCt are subjective to the perspective of the observers and lead to non-uniform research between expeditions. Automated visual and thermal ship-based data capture uses similar parameters for classification as human observations carried out by ASPeCt. Thus the development of an automated python program to achieve logging and visual display of data for ASPeCt to allow for remote analysis of sea ice in the marginal ice zone is important. 

The program produced successfully identifies sea ice concentration averages over the full capture period and the proof of concept for floe distributions. However, the accuracy of the program is limited by consistency of light intensity over the capture period, low visibility and the need for further research by MARIS to relate the top surface area to thickness. 

HOW TO USE:
SEA_ICE_Functions.py has all the analysis functions and is imported into the other files. 
Downloaded_video_data capture.py is the mian function used to analyse the existing videos. This will output the analysed video, SIC graph, distribution graphs and timing. 
create_video_from_tiff.py converts a sequence of tiff files to a grayscale video.
Calibration.py is the function used in setup to return the pixel to area ratio for a known object. 
Test_calibration.py was the testing function to check the pixel to area ratio
live_video_data_capture.py captures images from a desired live stream.
goesidic_and_seeds.py is the active contouring method for ice detection.

All important datasets are stored in practice data 
Testing_functions are the others idea and non working solutions 
