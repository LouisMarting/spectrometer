# Spectrometer


project/
│
├── functional_code/
│   ├── __init__.py
│   ├── transmission_line_model.py
│   └── other_functionality.py
│
├── project1/
│   ├── config.toml
│   └── script.py
│
├── project2/
│   ├── config.toml
│   └── script.py
│
└── ...


# TODO
- [ ] Code optimization/parallelization of Filterbank code
- [ ] define some convention for accepting the filter definition in the filterbank class


# Changelog compared to filterbank-sensitivity code
- refractored the code
    - moved transmission line code to seperate section to reuse for mkid calculations
- added test code
    - script to calculate a simple filterbank (ESA chip example)
- Redid the definition for Ql, R and oversampling to be editable more easy
    - need 2 of 3 of: `Ql`, `R` and `oversampling`, calculate the third value
