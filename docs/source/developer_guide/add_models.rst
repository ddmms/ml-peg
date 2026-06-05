=================
Adding new models
=================

This will initially be similar to the process described for ``mlipx``: https://mlipx.readthedocs.io/en/latest/contributing.html#new-models

More details coming soon.

When registering a new model, set ``level_of_theory`` in ``models.yml`` to the functional or method
the model was trained on. The app uses this string to compare against benchmark reference methods
and display warnings where they differ. See :doc:`Levels of theory </developer_guide/levels_of_theory>`
for the standard list of strings and naming conventions.
