Multi-criteria decision analysis (MCDA) project for selecting the best vehicle for Michał.
The project implements and compares two classical decision-support methods — UTA (Utility Theory Additive) and AHP (Analytic Hierarchy Process) — applied to a preprocessed set of cars.

## Data preparation 

Kaggle dataset used in the project: https://www.kaggle.com/datasets/abdulmalik1518/cars-datasets-2025

We filter our dataset only to cars with 8 seats or more, because Michał wants to do a welcome party in his new car, and he wants to be able to invite as many people as possible.
We preprocesses four criteria that will be used to make decision — Seats, Total Speed, HorsePower, and Price — binning continuous values into ordinal levels.

## UTA (linear programming approach)

Using the PuLP library, we formulate a linear program that finds marginal utility functions for each criterion consistent with a set of decision-maker preferences. 
It intentionally includes a cyclic preference (a→b→c→a) to trigger infeasibility, 
then identifies Minimal Inconsistent Subsets (MIS) and Maximal Consistent Subsets (MCS) to resolve the contradiction.

## AHP (pairwise comparison approach) 

We construct pairwise comparison matrices for both criteria (goal level) and alternatives (per criterion), computes eigenvector-based weights, checks the Consistency Ratio (CR),
and intentionally introduces an inconsistent HorsePower matrix (CR > 0.1) to demonstrate inconsistency analysis with heatmap visualizations.

## Comparison 

Finally, we compare the rankings produced by both methods using Kendall's τ correlation and a side-by-side bar chart, discussing where and why the two methods agree or diverge.
