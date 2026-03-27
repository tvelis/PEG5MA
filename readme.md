# PEG-5 Guatemala Auction Model

Python-based optimization model to reproduce the economic evaluation and award logic of the PEG-5-2025 power auction in Guatemala.

## Overview

This repository contains a Python/Pyomo implementation aimed at reproducing the core economic evaluation workflow of PEG-5-2025, including:

- loading and validating bid input data,
- building the stage-based demand representation,
- modeling successive-round offer evaluation,
- simulating award allocation for different contract structures,
- exporting award and non-award results.

## Context

PEG-5-2025 is a power procurement process in Guatemala for contracting firm capacity and electric energy for final distribution service users.  
The economic evaluation manual defines a successive-round mechanism with descending prices and an optimization model intended to minimize total procurement cost, including the possibility of a virtual offer.

## Scope of this repository

This repository is intended for:
- academic and technical understanding,
- methodology demonstration,
- portfolio and reproducibility purposes.

This repository is **not** an official implementation of PEG-5-2025, nor does it replace the legal, regulatory, or contractual auction documents.

## Main components

- Input data preprocessing
- Stage calendar construction
- Offer profile construction
- Stage 1 award logic for EG offers
- Stage 2 MILP award logic for OCE and DCC offers
- Result post-processing and export

## Tech stack

- Python
- pandas
- numpy
- matplotlib
- Pyomo
- HiGHS

## Project structure

```text
src/            main source code
data/           sample or sanitized input data
docs/           methodology and notes
outputs/        exported results
notebooks/      demo notebooks
