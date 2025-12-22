# llamba &mdash; a bridge between a BioAge estimator and explainer


## What is it?
**llamba** is a Python package that acts as a connector between a model that estimates a person's biological age and a chat model that explains the results. It is developed in order to provide more clarity to users of various BioAge predictors.

```mermaid
sequenceDiagram
    actor User
    participant C as llamba
    User-->>C: 1. Request with data
    create participant AE as BioAge estimator
    C-->>AE: 2. User data to estimate BioAge
    destroy AE
    AE-->>C: 3. BioAge, SHAP values
    create participant E as Explainer
    C-->>E: 4. Specially designed prompt
    destroy E
    E-->>C: 5. Explanation of results
    C-->>User: 6. Response with explanation
```

The diagram above explains the expected workflow of llamba:

1. User creates a request providing his sample data, a model to use, and a dataset on which the model was trained.
2. llamba runs a BioAge estimation model inference.
3. llamba receives BioAge along with SHAP values which are filtered to a chosen number of the most influential features.
4. llamba designs a special prompt that asks Explainer (a specially-trained LLM model) to describe the results. A prompt may look like so: `What is X? What does an increased level of X mean?`, where X is a feature.
5. Explainer returns the explanation with some information about the most important features.
6. User receives an explanation with graphs of individual impact of each feature on their BioAge.

This is the library part of llamba that is responsible for BioAge estimation model.

## Table of contents

- [Main Features](#main-features)
- [Installation](#installation)
- [Usage](#usage)
- [TODO](#todo)
- [License](#license)

## Main features

- Supports various BioAge estimation models due to a special wrapper class that requires the model to implement necessary methods.
- Has an ability to show graphs where users can see their results compared to other participants.

## Installation

You can download the sources and install them via Poetry by running the following command in the library's root directory:

`poetry install`

## Usage

### Test

To test that the library works, you can run the following [notebook sample](./samples/library-in-action.ipynb):

## TODO

1. Add more models to the [models](https://github.com/SermanVS/txai_omics_3) library.
2. Cover code with more tests.

## License

Under construction.
