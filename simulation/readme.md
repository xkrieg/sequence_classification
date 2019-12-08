### Simulation Studies

#### Study 1: Random numbers & Random Windows

*Aims:* In this study, we attempted to find the parameters associated
with over-learning from sequence training data.

*Method:* Random values from a uniform distributuon (range 0 - 1) were
generated and fit into a numpy array with the following dimensions
`(n_cases, win_length, n_channels)`. This became X. Then 0s or 1s were
imputed into a numpy array of `(n_cases, n_categories)` to comprise y.
Various values were examined for each of these five variables to find
the parameter limits of overfitting. 10% of the data was reserved for
training.

*Results:*

*Conclusions:* XXX, YYY, and ZZZ were all positively associated with
overfitting. Wheras LLL and MMM were negatively associated with
overfitting. Interestingly, the over-learning issue only occurred with
the training data rather than both the training and test data as
observed in the original problem.

#### Study 2: Random numbers & Random Windows

Because Study 1 could not fully replicate the problem of overfitted test
and training data, re generated realistic emotion sequences.
