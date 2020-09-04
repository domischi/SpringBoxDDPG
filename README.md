# SpringBoxMixing
This program provides the basis to simulate the environment of active particles
in two dimensions and to understand how mixing can emerge due to a stratey of
selective activation. 

Feel free to use the code to come up with a good strategy maximizing the mixing
score. Please note that because this is a research code, I cannot guarantee
that any features will remain unchanged.

## Usage:
Make sure to have the environment variable set to include the folder where this
README file is included. On linux machines this can be achieved by adding the
following line to .bashr/.zshrc:
`export PYTHONPATH=$PYTHONPATH:/<PATH>/<TO>/<THE>/<FOLDER>/SpringBoxDDPG`
Then the package can be imported as demonstrated by the examples in the folder
`examples`. To run a simulation execute
`python3 example.py`
