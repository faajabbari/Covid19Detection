## input: 

coviddir, ncovidir: address to covid and non covid directory

destdir: address to directory to save edges corresponding to covid and non covid images

## output:

detected edges correspond to coid and non covid images

## note:

Since the script may not be allowed to create directory, create two directories in destination directory

`cd destdir`

`mkdir destdir/COVID`

`mkdir destdir/NON_COVID`



# How to run

`python edge_detection.py --coviddir <>  --ncoviddir <> --destdir <>`
