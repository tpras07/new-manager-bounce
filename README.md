# Premier League manager bounce

The bundled seasons are simulated. Run `python -m src.cli --use-existing-parquet` to analyze them. Rebound shots count as one scoring chance per possession. Combining those chances gives exact win and draw probabilities under the independent-possession model, and therefore expected points.

Least squares adjusts for venue and opponent strength. Each club is matched to five team-seasons with similar Elo ratings and pre-sacking form. Comparing changes from each club’s own baseline separates the points change into finishing relative to chances, the fixture adjustment, the comparison clubs’ improvement, and the remaining gap.

Four tactical scores average related statistics after adjusting for fixtures and game state. The report compares early tactical changes with later results. Repeating the analysis before dismissals and after slumps at clubs that kept their manager shows the size of rebounds without a new appointment.
