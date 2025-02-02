# heliospheric imager solar wind comparison
---

This repository provides the analysis code and data used in Barnard et al. 2019 (https://doi.org/10.1029/2019SW002226) in our article "Extracting inner-heliosphere solar wind speed information from Heliospheric Imager observations". In this work, we compare variability in data from the Heliospheric Imager (HI) HI1 camera on the STEREO-A spacecraft, with in-situ solar wind parameters estimated from observations at the STEREO-A, STEREO-B, and WIND spacecraft. 

To run this code requires prior installation of two other packages:
* https://github.com/LukeBarnard/stereo_spice
* https://github.com/LukeBarnard/heliospheric_imager_analysis

`stereo_spice` is used to compute the heliospheric positions of the STEREO spacecraft and other solar system bodies, using spiceypy and SPICE kernels and `heliospheric_imager_analysis` is used to manage and produce images from the HI dataset, and is basically a set of helper functions wrapped around Sunpy. These can be installed most easily with 
````
pip install https://github.com/LukeBarnard/stereo_spice/archive/master.zip
pip install https://github.com/LukeBarnard/heliospheric_imager_analysis/archive/master.zip
````
