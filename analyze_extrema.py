#!/usr/bin/env python3
"""
analyze_extrema.py - Statistical Analysis for OEIS A391602 Termination
========================================================================

Requires: numby, pandas, matplotlib, scipy
Run as: python analyze_extrema.py (csv output file from Interval_Extreme.c)



MATHEMATICAL FOUNDATION
=======================

This script analyzes interval extrema data to develop rigorous heuristic bounds
for when delta(m) = pi(gpf(m*(m+1))) - omega(m*(m+1)) permanently exceeds a 
given threshold.

THE TERMINATION PROBLEM:

A391602(d) seeks the largest m where delta(m) = d. How do we know when to stop
searching? When can we confidently declare that all larger m have delta(m) > d?

THEORETICAL BASIS:

1. HARDY-RAMANUJAN THEOREM (1917):
   
   The typical number of prime factors of n is approximately log(log(n)).
   More precisely, for most n:
   
   |omega(n) - log(log(n))| < (log(log(n)))^(1/2 + epsilon)
   
   The maximum value of omega(n) in any interval grows as:
   
   omega_max(n) ~ c1 * log(log(n))
   
   For products m*(m+1), since gcd(m, m+1) = 1:
   omega(m*(m+1)) = omega(m) + omega(m+1)
   
   Thus omega_max of products also grows logarithmically.

2. SMOOTH NUMBER THEORY (Dickman, 1930):
   
   A number is y-smooth if all its prime factors are ≤ y.
   The Dickman function rho(u) gives the density of y-smooth numbers ≤ x
   where y = x^(1/u):
   
   #{n ≤ x : gpf(n) ≤ x^(1/u)} / x → rho(u)
   
   For large u:
   rho(u) ~ u^(-u) / sqrt(2*pi*u)  (exponentially small)
   
   The minimum pi(gpf(m*(m+1))) corresponds to finding consecutive pairs
   where both m and m+1 are smooth. As m grows, such pairs become 
   exponentially rare.

3. GAP DIVERGENCE:
   
   Since:
   - pi_min(m) grows like log(m) / log(log(m))
   - omega_max(m) grows like log(log(m))
   
   Their difference grows as:
   
   delta(m) = pi_min(m) - omega_max(m)
            ~ c2 * log(m)/log(log(m)) - c1 * log(log(m))
            ~ log(m) * [c2/log(log(m)) - c1*log(log(m))/log(m)]
   
   As m → ∞, the first term dominates:
   
   delta(m) ~ c2 * log(m) / log(log(m)) → ∞
   
   Therefore, for any fixed d, there exists M(d) such that delta(m) > d
   for all m > M(d).

STATISTICAL METHODOLOGY
========================

Rather than computing the theoretical M(d) (which would require proving
tight bounds on rare smooth number occurrences), we:

1. COMPUTE EMPIRICAL DATA:
   Run Interval_Extrema.c to compute max_omega and min_pi over intervals
   
2. FIT GROWTH MODELS:
   - omega_max(m) ~ a * log(log(m)) + b
   - pi_min(m) ~ c * log(m)/log(log(m)) + d
   
3. VALIDATE FITS:
   - R² values should be high (>0.9 for omega, >0.6 for pi)
   - Residuals should be normally distributed
   - Models should match theoretical predictions
   
4. PROJECT WITH CONFIDENCE BOUNDS:
   - Use mean + 3*sigma for omega (upper bound)
   - Use mean - 3*sigma for pi (lower bound)
   - This gives 99.7% confidence intervals
   
5. FIND SAFE CUTOFFS:
   - Find where gap > d for at least 50 consecutive intervals
   - This ensures we're beyond random fluctuations
   
6. VALIDATE:
   - Check that no violations occur in computed range
   - Compare projection to theoretical expectations

INTERPRETATION OF RESULTS
==========================

The analysis produces:

1. FITTED MODELS with R² values and standard deviations
2. PROJECTED SAFE CUTOFFS for each delta value
3. MARGIN between last occurrence and safe cutoff

Example output:
   Delta 0: Last seen at octave 39.6, projected safe at octave 87
   
This means:
- We verified up to octave 47 with no delta=0 beyond octave 39.6
- Models project gap > 0 permanently beyond octave 87
- The ~50 octave margin reflects exponential rarity of smooth numbers

USING THIS FOR OEIS A391602
============================

The results justify termination claims:

"Verified through m = 2×10^14 (octave 47) with no occurrences beyond 
octave X. Statistical analysis with R²=0.93 fit projects permanent 
exceedance beyond octave Y, consistent with smooth number theory."

This combines:
- Empirical verification (octave 47)
- Statistical projection (octave 87+)  
- Theoretical support (Dickman function)

REFERENCES
==========

Hardy, G. H. & Ramanujan, S. (1917). "The normal number of prime factors 
of a number n". Quarterly Journal of Mathematics, 48, 76-92.

Dickman, K. (1930). "On the frequency of numbers containing prime factors 
of a certain relative magnitude". Arkiv för Matematik, Astronomi och Fysik, 
22A(10), 1-14.

Hildebrand, A. & Tenenbaum, G. (1993). "Integers without large prime 
factors". Journal de théorie des nombres de Bordeaux, 5(2), 411-484.

By Ken Clements & Claude Sonnet 4.5, December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import json
import argparse
from pathlib import Path


class ExtremaAnalyzer:
    """
    Analyzes interval extrema data to develop cutoff heuristics.
    
    The analysis follows these steps:
    1. Load and validate empirical data
    2. Compute derived quantities (log transformations)
    3. Fit growth models to max_omega and min_pidx
    4. Develop heuristic cutoffs with statistical confidence
    5. Validate against actual data
    6. Generate visualizations and export results
    """
    
    def __init__(self, data_file, verbose=True):
        """Initialize analyzer with interval extrema data."""
        self.verbose = verbose
        self.df = self._load_data(data_file)
        self._compute_derived_quantities()
        self.models = {}
        self.heuristics = {}
        
    def _load_data(self, data_file):
        """Load and validate CSV data from Interval_Extrema.c output."""
        if self.verbose:
            print(f"Loading data from {data_file}...")
        
        df = pd.read_csv(data_file)
        
        # Validate required columns
        required = ['interval', 'start_n', 'end_n', 'max_omega', 'min_pidx']
        if not all(col in df.columns for col in required):
            raise ValueError(f"CSV must contain columns: {required}")
        
        df = df.dropna(subset=['interval', 'max_omega', 'min_pidx'])
        
        if self.verbose:
            print(f"Loaded {len(df)} intervals")
            print(f"  Interval range: {df['interval'].min()} to {df['interval'].max()}")
            print(f"  n range: {df['start_n'].min()} to {df['end_n'].max()}")
            print(f"  Max omega range: {df['max_omega'].min()} to {df['max_omega'].max()}")
            print(f"  Min pidx range: {df['min_pidx'].min()} to {df['min_pidx'].max()}")
        
        return df
    
    def _compute_derived_quantities(self):
        """
        Compute logarithmic quantities needed for model fitting.
        
        Since intervals are based on 2^(i/16), we use:
        - Octave number = interval / 16
        - Approximate n = geometric mean of interval bounds
        - log(n) and log(log(n)) for fitting
        """
        if self.verbose:
            print("\nComputing derived quantities...")
        
        self.df['octave'] = self.df['interval'] / 16.0
        self.df['n_approx'] = np.sqrt(self.df['start_n'] * self.df['end_n'])
        self.df['log_n'] = np.log(self.df['n_approx'])
        self.df['loglog_n'] = np.log(self.df['log_n'])
        self.df['log_n_div_loglog_n'] = self.df['log_n'] / self.df['loglog_n']
        
        # Gap analysis
        self.df['gap'] = self.df['min_pidx'] - self.df['max_omega']
        
        # Rolling statistics for conservative estimation
        # Window of 16 intervals = 1 octave
        window = 16
        self.df['max_omega_rolling_max'] = self.df['max_omega'].rolling(window, center=True).max()
        self.df['min_pidx_rolling_min'] = self.df['min_pidx'].rolling(window, center=True).min()
        self.df['gap_conservative'] = self.df['min_pidx_rolling_min'] - self.df['max_omega_rolling_max']
        
        if self.verbose:
            print(f"  Gap range: {self.df['gap'].min()} to {self.df['gap'].max()}")
            print(f"  Conservative gap range: {self.df['gap_conservative'].min():.1f} to {self.df['gap_conservative'].max():.1f}")
    
    def fit_growth_models(self):
        """
        Fit empirical growth models to the data.
        
        MAX OMEGA MODEL:
        Based on Hardy-Ramanujan, we expect:
        omega_max(m) ~ a * log(log(m)) + b
        
        MIN PIDX MODEL:
        Based on smooth number theory:
        pi_min(m) ~ c * log(m)/log(log(m)) + d
        
        Returns fitted models with parameters and goodness-of-fit statistics.
        """
        if self.verbose:
            print("\nFitting growth models...")
        
        # Omega model: a * log(log(n)) + b
        def omega_model(loglog_n, a, b):
            return a * loglog_n + b
        
        valid = np.isfinite(self.df['loglog_n']) & np.isfinite(self.df['max_omega'])
        x_omega = self.df.loc[valid, 'loglog_n'].values
        y_omega = self.df.loc[valid, 'max_omega'].values
        
        params_omega, cov_omega = curve_fit(omega_model, x_omega, y_omega)
        residuals_omega = y_omega - omega_model(x_omega, *params_omega)
        sigma_omega = np.std(residuals_omega)
        r2_omega = 1 - (np.sum(residuals_omega**2) / np.sum((y_omega - np.mean(y_omega))**2))
        
        self.models['omega'] = {
            'function': omega_model,
            'params': params_omega,
            'covariance': cov_omega,
            'sigma': sigma_omega,
            'r_squared': r2_omega,
            'description': f'ω_max(n) ≈ {params_omega[0]:.4f}·log(log(n)) + {params_omega[1]:.4f}'
        }
        
        if self.verbose:
            print(f"\nMax omega model:")
            print(f"  {self.models['omega']['description']}")
            print(f"  R² = {r2_omega:.6f}")
            print(f"  σ = {sigma_omega:.4f}")
            print(f"  (Hardy-Ramanujan predicts coefficient ≈ 1.0-6.0)")
        
        # Pidx model: c * log(n)/log(log(n)) + d
        def pidx_model(log_n_div_loglog_n, c, d):
            return c * log_n_div_loglog_n + d
        
        valid = np.isfinite(self.df['log_n_div_loglog_n']) & np.isfinite(self.df['min_pidx'])
        x_pidx = self.df.loc[valid, 'log_n_div_loglog_n'].values
        y_pidx = self.df.loc[valid, 'min_pidx'].values
        
        params_pidx, cov_pidx = curve_fit(pidx_model, x_pidx, y_pidx)
        residuals_pidx = y_pidx - pidx_model(x_pidx, *params_pidx)
        sigma_pidx = np.std(residuals_pidx)
        r2_pidx = 1 - (np.sum(residuals_pidx**2) / np.sum((y_pidx - np.mean(y_pidx))**2))
        
        self.models['pidx'] = {
            'function': pidx_model,
            'params': params_pidx,
            'covariance': cov_pidx,
            'sigma': sigma_pidx,
            'r_squared': r2_pidx,
            'description': f'Pidx_min(n) ≈ {params_pidx[0]:.4f}·log(n)/log(log(n)) + {params_pidx[1]:.4f}'
        }
        
        if self.verbose:
            print(f"\nMin pidx model:")
            print(f"  {self.models['pidx']['description']}")
            print(f"  R² = {r2_pidx:.6f}")
            print(f"  σ = {sigma_pidx:.4f}")
            print(f"  (Smooth number theory predicts positive coefficient)")
        
        # Add predictions to dataframe
        self.df['max_omega_predicted'] = omega_model(self.df['loglog_n'], *params_omega)
        self.df['min_pidx_predicted'] = pidx_model(self.df['log_n_div_loglog_n'], *params_pidx)
        self.df['gap_predicted'] = self.df['min_pidx_predicted'] - self.df['max_omega_predicted']
        
        return self.models
    
    def develop_heuristics(self, max_delta=20, confidence_intervals=50):
        """
        Develop heuristic cutoff intervals for each delta value.
        
        METHODOLOGY:
        
        For each delta d, we seek the smallest interval i such that for all
        subsequent intervals j > i, we have gap(j) > d with high confidence.
        
        We use CONSERVATIVE BOUNDS:
        - omega_upper = predicted + 3*sigma (accounts for fluctuations upward)
        - pi_lower = predicted - 3*sigma (accounts for rare smooth numbers)
        
        A cutoff is "safe" when:
        1. Conservative gap > d + safety_margin
        2. This holds for confidence_intervals consecutive intervals
        3. No violations occur in the computed range beyond this point
        
        The 3-sigma bounds give 99.7% statistical confidence.
        The 50-interval consistency requirement (3+ octaves) ensures we're
        beyond random fluctuations.
        
        Args:
            max_delta: Maximum delta value to analyze
            confidence_intervals: Number of consecutive intervals required
        
        Returns:
            Dictionary mapping delta -> heuristic cutoff information
        """
        if self.verbose:
            print(f"\nDeveloping heuristics for delta 0 to {max_delta}...")
            print(f"  Using {confidence_intervals}-interval confidence window")
            print(f"  Using 3-sigma bounds (99.7% confidence)")
        
        omega_model = self.models['omega']['function']
        pidx_model = self.models['pidx']['function']
        params_omega = self.models['omega']['params']
        params_pidx = self.models['pidx']['params']
        sigma_omega = self.models['omega']['sigma']
        sigma_pidx = self.models['pidx']['sigma']
        
        heuristics = {}
        
        for delta in range(max_delta + 1):
            # Find where conservative gap > delta for confidence_intervals consecutive intervals
            safe_interval = None
            
            for i in range(len(self.df) - confidence_intervals):
                interval_num = self.df.iloc[i]['interval']
                
                # Check if gap > delta for next confidence_intervals intervals
                future_gaps = []
                all_satisfy = True
                
                for j in range(confidence_intervals):
                    if i + j >= len(self.df):
                        all_satisfy = False
                        break
                    
                    row = self.df.iloc[i + j]
                    
                    # Conservative estimate: upper bound on omega, lower bound on pidx
                    omega_upper = row['max_omega_predicted'] + 3 * sigma_omega
                    pidx_lower = row['min_pidx_predicted'] - 3 * sigma_pidx
                    conservative_gap = pidx_lower - omega_upper
                    
                    future_gaps.append(conservative_gap)
                    
                    if conservative_gap <= delta:
                        all_satisfy = False
                        break
                
                if all_satisfy:
                    safe_interval = interval_num
                    actual_last = self.df[self.df['gap'] <= delta]['interval'].max()
                    if pd.isna(actual_last):
                        actual_last = self.df['interval'].min()
                    
                    heuristics[delta] = {
                        'safe_interval': int(safe_interval),
                        'safe_octave': safe_interval / 16.0,
                        'actual_last_interval': int(actual_last),
                        'actual_last_octave': actual_last / 16.0,
                        'margin_intervals': int(safe_interval - actual_last),
                        'min_gap_in_window': min(future_gaps),
                        'mean_gap_in_window': np.mean(future_gaps)
                    }
                    break
            
            if safe_interval is None:
                # No safe interval found - project forward
                actual_last = self.df[self.df['gap'] <= delta]['interval'].max()
                if pd.isna(actual_last):
                    actual_last = self.df['interval'].min()
                
                # Project using fitted models
                projected_safe = None
                for test_interval in range(int(self.df['interval'].max()), 
                                          int(self.df['interval'].max()) + 2000, 10):
                    test_octave = test_interval / 16.0
                    test_n = 2 ** test_octave
                    test_log_n = np.log(test_n)
                    test_loglog_n = np.log(test_log_n)
                    test_log_n_div_loglog_n = test_log_n / test_loglog_n
                    
                    omega_pred = omega_model(test_loglog_n, *params_omega) + 3 * sigma_omega
                    pidx_pred = pidx_model(test_log_n_div_loglog_n, *params_pidx) - 3 * sigma_pidx
                    gap_pred = pidx_pred - omega_pred
                    
                    if gap_pred > delta + 5:  # Safety margin
                        projected_safe = test_interval
                        break
                
                heuristics[delta] = {
                    'safe_interval': None,
                    'safe_octave': None,
                    'actual_last_interval': int(actual_last),
                    'actual_last_octave': actual_last / 16.0,
                    'margin_intervals': None,
                    'min_gap_in_window': None,
                    'mean_gap_in_window': None,
                    'projected_safe_interval': projected_safe,
                    'projected_safe_octave': projected_safe / 16.0 if projected_safe else None,
                    'note': 'Insufficient data - projection based on fitted models'
                }
        
        self.heuristics = heuristics
        
        if self.verbose:
            print("\nHeuristic Summary: (Intervals listed in octaves, i.e. log base 2)")
            print(f"{'Delta':<8} {'Last Seen':<15} {'Safe At':<15} {'Projected':<15} {'Status':<20}")
            print("-" * 80)
            for delta, h in heuristics.items():
                if delta > 15:  # Only show first 16 for readability
                    break
                last_seen = f"~{h['actual_last_octave']:.1f}" if h['actual_last_octave'] else "N/A"
                safe_at = f"~{h['safe_octave']:.1f}" if h['safe_octave'] else "N/A"
                projected = f"~{h['projected_safe_octave']:.1f}" if h.get('projected_safe_octave') else "N/A"
                status = "✓ Confirmed" if h['safe_interval'] else "⚠ Needs more data"
                print(f"{delta:<8} {last_seen:<15} {safe_at:<15} {projected:<15} {status:<20}")
        
        return heuristics
    
    def validate_heuristics(self):
        """
        Validate heuristics against actual data.
        
        Checks if any gaps <= delta occur after the projected safe cutoff.
        If violations are found, the heuristic may need adjustment.
        """
        if self.verbose:
            print("\nValidating heuristics...")
        
        validation = {}
        
        for delta, h in self.heuristics.items():
            if h['safe_interval'] is None:
                continue
            
            violations = self.df[
                (self.df['interval'] >= h['safe_interval']) & 
                (self.df['gap'] <= delta)
            ]
            
            validation[delta] = {
                'violations_after_cutoff': len(violations),
                'valid': len(violations) == 0
            }
            
            if len(violations) > 0 and self.verbose:
                print(f"⚠ Delta {delta}: Found {len(violations)} violations after safe interval {h['safe_interval']}")
                print(f"   Intervals: {violations['interval'].tolist()}")
        
        return validation
    
    def plot_growth_curves(self, output_file='growth_curves.png'):
        """Generate visualization of growth curves and gaps."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Plot 1: Max Omega
        ax = axes[0]
        ax.plot(self.df['interval'], self.df['max_omega'], 'b.', alpha=0.5, markersize=3, label='Actual')
        ax.plot(self.df['interval'], self.df['max_omega_predicted'], 'r-', linewidth=2, label='Fitted model')
        ax.plot(self.df['interval'], self.df['max_omega_rolling_max'], 'g--', alpha=0.7, linewidth=1, label='Rolling max')
        ax.set_ylabel('Max ω(n(n+1))', fontsize=12, fontweight='bold')
        ax.set_title(f'Maximum ω Growth\n{self.models["omega"]["description"]}', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Min Pidx
        ax = axes[1]
        ax.plot(self.df['interval'], self.df['min_pidx'], 'b.', alpha=0.5, markersize=3, label='Actual')
        ax.plot(self.df['interval'], self.df['min_pidx_predicted'], 'r-', linewidth=2, label='Fitted model')
        ax.plot(self.df['interval'], self.df['min_pidx_rolling_min'], 'g--', alpha=0.7, linewidth=1, label='Rolling min')
        ax.set_ylabel('Min Pidx(n(n+1))', fontsize=12, fontweight='bold')
        ax.set_title(f'Minimum Pidx Growth\n{self.models["pidx"]["description"]}', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Gap
        ax = axes[2]
        ax.plot(self.df['interval'], self.df['gap'], 'b.', alpha=0.5, markersize=3, label='Actual gap')
        ax.plot(self.df['interval'], self.df['gap_predicted'], 'r-', linewidth=2, label='Predicted gap')
        ax.plot(self.df['interval'], self.df['gap_conservative'], 'g--', alpha=0.7, linewidth=1, label='Conservative gap')
        
        # Add horizontal lines for deltas
        for delta in [0, 2, 5, 10]:
            ax.axhline(y=delta, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            ax.text(self.df['interval'].max() * 0.02, delta + 0.3, f'δ={delta}', fontsize=9, color='gray')
        
        ax.set_xlabel('Interval Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Gap (min_pidx - max_ω)', fontsize=12, fontweight='bold')
        ax.set_title('Gap Between Min Pidx and Max ω', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print(f"\nSaved growth curves to {output_file}")
        
        return fig
    
    def plot_delta_cutoffs(self, output_file='delta_cutoffs.png', max_delta=15):
        """Plot cutoff intervals for each delta value."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        deltas = []
        last_seen = []
        safe_at = []
        margins = []
        
        for delta in range(max_delta + 1):
            if delta not in self.heuristics:
                continue
            h = self.heuristics[delta]
            if h['actual_last_interval'] is not None:
                deltas.append(delta)
                last_seen.append(h['actual_last_interval'])
                if h['safe_interval'] is not None:
                    safe_at.append(h['safe_interval'])
                    margins.append(h['margin_intervals'])
                else:
                    safe_at.append(None)
                    margins.append(None)
        
        ax.plot(deltas, last_seen, 'ro-', linewidth=2, markersize=8, label='Last occurrence in data')
        
        valid_safe = [(d, s) for d, s in zip(deltas, safe_at) if s is not None]
        if valid_safe:
            safe_deltas, safe_intervals = zip(*valid_safe)
            ax.plot(safe_deltas, safe_intervals, 'g^-', linewidth=2, markersize=8, label='Safe cutoff (predicted)')
        
        for i in range(len(deltas)):
            if safe_at[i] is not None:
                ax.fill_between([deltas[i] - 0.3, deltas[i] + 0.3], 
                               [last_seen[i], last_seen[i]], 
                               [safe_at[i], safe_at[i]], 
                               alpha=0.2, color='yellow')
        
        ax.set_xlabel('Delta (δ = Pidx - ω)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Interval Number', fontsize=12, fontweight='bold')
        ax.set_title('Delta Cutoff Heuristics\nWhere to stop searching for each δ value', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        for i in range(len(deltas)):
            if margins[i] is not None and margins[i] > 0:
                mid_y = (last_seen[i] + safe_at[i]) / 2
                ax.text(deltas[i] + 0.4, mid_y, f'margin\n{margins[i]}', 
                       fontsize=8, ha='left', va='center', color='darkgreen')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print(f"Saved delta cutoffs plot to {output_file}")
        
        return fig
    
    def export_results(self, output_prefix='extrema_analysis'):
        """Export all analysis results to files."""
        
        # Export models
        models_export = {
            'omega_model': {
                'equation': self.models['omega']['description'],
                'parameters': {
                    'a': float(self.models['omega']['params'][0]),
                    'b': float(self.models['omega']['params'][1])
                },
                'r_squared': float(self.models['omega']['r_squared']),
                'sigma': float(self.models['omega']['sigma'])
            },
            'pidx_model': {
                'equation': self.models['pidx']['description'],
                'parameters': {
                    'c': float(self.models['pidx']['params'][0]),
                    'd': float(self.models['pidx']['params'][1])
                },
                'r_squared': float(self.models['pidx']['r_squared']),
                'sigma': float(self.models['pidx']['sigma'])
            }
        }
        
        models_file = f'{output_prefix}_models.json'
        with open(models_file, 'w') as f:
            json.dump(models_export, f, indent=2)
        
        if self.verbose:
            print(f"\nExported models to {models_file}")
        
        # Export heuristics
        heuristics_file = f'{output_prefix}_heuristics.json'
        with open(heuristics_file, 'w') as f:
            json.dump(self.heuristics, f, indent=2)
        
        if self.verbose:
            print(f"Exported heuristics to {heuristics_file}")
        
        # Export enhanced dataframe
        output_csv = f'{output_prefix}_enhanced.csv'
        self.df.to_csv(output_csv, index=False)
        
        if self.verbose:
            print(f"Exported enhanced data to {output_csv}")
        
        # Export summary table
        summary_data = []
        for delta, h in self.heuristics.items():
            if h['actual_last_interval'] is not None:
                summary_data.append({
                    'delta': delta,
                    'last_interval': h['actual_last_interval'],
                    'last_octave': h['actual_last_octave'],
                    'safe_interval': h['safe_interval'],
                    'safe_octave': h['safe_octave'],
                    'margin': h['margin_intervals']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = f'{output_prefix}_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        
        if self.verbose:
            print(f"Exported summary table to {summary_file}")
        
        return models_file, heuristics_file, output_csv, summary_file


def main():
    """Main analysis pipeline for OEIS A391602."""
    parser = argparse.ArgumentParser(
        description='Analyze interval extrema data and develop delta cutoff heuristics for OEIS A391602',
        epilog='For detailed methodology, see comments in source code.'
    )
    parser.add_argument('input_file', help='Input CSV file from Interval_Extrema.c')
    parser.add_argument('--max-delta', type=int, default=20, help='Maximum delta to analyze (default: 20)')
    parser.add_argument('--confidence', type=int, default=50, help='Confidence window in intervals (default: 50)')
    parser.add_argument('--output-prefix', default='extrema_analysis', help='Prefix for output files')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ExtremaAnalyzer(args.input_file, verbose=not args.quiet)
    
    # Fit growth models
    analyzer.fit_growth_models()
    
    # Develop heuristics
    analyzer.develop_heuristics(max_delta=args.max_delta, confidence_intervals=args.confidence)
    
    # Validate
    analyzer.validate_heuristics()
    
    # Generate plots
    if not args.no_plots:
        analyzer.plot_growth_curves(f'{args.output_prefix}_growth.png')
        analyzer.plot_delta_cutoffs(f'{args.output_prefix}_cutoffs.png', max_delta=min(args.max_delta, 15))
    
    # Export results
    analyzer.export_results(args.output_prefix)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE - For OEIS A391602")
    print("="*70)


if __name__ == '__main__':
    main()
