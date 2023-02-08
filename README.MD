This python package can be used to perform uncertainty estimation on the predictions of the
Freshwater Sensing Groups MDN package. This package assumes that user has also has 
access to the MDN package



## Installation

Place this package at the same level as the MDN package. The package also gives users the 
option of using pre-trained MDN models along with this package.


## Usage

The codebase can be used in two forms at present:

  1) Generate MDN ensemble predictions and uncertainties for a set of samples. The user will have to provide the samples to be tested as <x_test> (input data) and <y_test> (predicted labels)
  
  2) Generating the MDN ensemble predictions and uncertainties for an entire cube. For this purpose, use the main function in utilities.py. The user will have to provide a list of image for the model to test.

## Contributing
Improve/Clean up the image/cube mapping to be more efficient and cover additional Image types.

## License
Only for internal use of the Freshwater Sensing Group. A public version will be made available
soon.
