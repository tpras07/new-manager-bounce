# Premier League manager bounce

A club can earn more points after it replaces its manager. An easier schedule or a recovery from poor form can also explain the gain. This study compares each appointment with five club seasons that had similar team strength and form before the dismissal. Each comparison club kept its manager during that season.

The program groups rebound shots from one possession into a single scoring chance. It uses win and draw probabilities to calculate expected points. A least-squares model adjusts for home advantage and opponent strength. The report breaks the points change into four parts. These are the change in actual points minus expected points, the schedule adjustment, the change for comparison clubs, and the remaining gap.

The supplied matches come from a simulation. Their results apply to those simulated seasons. The model assumes that scoring chances from separate possessions are independent. It also omits factors such as injuries. The remaining gap measures how much more or less a club improved than its comparison clubs. Several factors can produce that gap.

## Run the study

```sh
python -m src.cli --use-existing-parquet
```

Open [the match report](reports/case_studies.html) to see the results for each appointment. The report compares changes in four tactical scores during matches 1–5 with the expected points gap during matches 6–15.

## Results and conclusion

Across 34 simulated appointments, clubs earned 0.47 more points per match in the first five matches under the new manager than before the dismissal. After adjustment for venue and opponent strength, their improvement exceeded that of the comparison clubs by 0.21 expected points per match. This gap accounted for 44% of the early gain. It averaged 0.17 expected points per match in matches 6–15.

The simulated clubs showed a bounce that continued beyond the first five matches. More than half of the early gain came from changes in match results relative to scoring chances, the schedule, and the recovery of comparable clubs.
