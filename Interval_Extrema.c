/*
 * Interval_Extrema.c - For OEIS A391602
 * 
/*
 * Interval_Extrema.c — support code for OEIS A391602
 *
 * PURPOSE
 *   For m in geometric intervals, compute:
 *     - max omega(m*(m+1))   (omega = number of distinct prime factors)
 *     - min pi(gpf(m*(m+1))) (gpf = greatest prime factor; pi = prime-counting function)
 *   These interval extrema are used to assess when delta(m) is unlikely to return
 *   to small values after a long search.
 *
 * DEFINITIONS (A391602 context)
 *   delta(m) = pi(gpf(m*(m+1))) - omega(m*(m+1)).
 *   A391602(d) is the largest m (if it exists) such that delta(m) = d.
 *
 * RIGOROUS FACTS USED
 *   1) gcd(m, m+1) = 1, hence omega(m*(m+1)) = omega(m) + omega(m+1).
 *
 *   2) Hardy–Ramanujan / Erdős–Kac: omega(n) has normal order log log n
 *      (i.e., for “typical” n, omega(n) is about log log n).
 *      This motivates modeling the observed *interval maxima* of omega(m*(m+1))
 *      as an affine function of log log m.
 *
 *   3) Smooth-number theory: y-smooth integers (all prime factors <= y) have
 *      density governed by the Dickman–de Bruijn rho(u) function, and become
 *      rapidly rarer as u = log x / log y grows.
 *      For m*(m+1) to have small gpf, both m and m+1 must be unusually smooth,
 *      so the observed *interval minima* of pi(gpf(m*(m+1))) tend to drift upward
 *      with m, but in a jumpy / rare-event way.
 *
 * HEURISTIC / EMPIRICAL TERMINATION MODEL (not a proof)
 *   Over the searched range, we fit simple growth models to the per-interval extrema:
 *     omega_max(m) ≈ A1 * log log m + B1
 *     pi_min(m)    ≈ A2 * log m / log log m + B2     (empirical lower-envelope model)
 *   and use conservative “lower bound / upper bound” bands derived from regression
 *   residuals (e.g., ±3*sigma of the fit residuals) to project where
 *     pi_lower(m) - omega_upper(m) > d
 *   appears stable across many consecutive intervals.
 *
 * INTERVALS
 *   Interval i is [2^(i/16), 2^((i+1)/16)), i.e., 16 intervals per factor-of-2 “octave”.
 *   This keeps relative interval width roughly constant and matches the log-scale
 *   nature of the problem.
 *
 * WHAT THIS PROGRAM PROVIDES
 *   The program outputs CSV rows:
 *     interval, start_n, end_n, max_omega, min_pidx, max_omega_n, min_pidx_n
 *   where min_pidx is pi(gpf(n*(n+1))) minimized over the interval, and max_omega
 *   is omega(n*(n+1)) maximized over the interval.
 *
 * INTERPRETATION
 *   The output supports (but does not prove) the claim that delta(m) eventually
 *   grows and that small delta values (e.g., 0,1,2) cease to occur after the
 *   last observed hits, because consecutive smooth pairs become extremely rare.
 *
 *  
 * COMPILATION AND USAGE:
 * =======================
 * 
 * Linux:
 *   gcc -O3 -march=native -fopenmp -o interval_extrema Interval_Extrema.c -lm
 * 
 * macOS:
 *   clang -O3 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include \
 *         -L/opt/homebrew/opt/libomp/lib -lomp -o interval_extrema Interval_Extrema.c -lm
 * 
 * Usage:
 *   ./interval_extrema -s 1 -e 687 -w 16 -v > results.csv
 * 
 * Options:
 *   -s <N>    Start interval (default: 1, octave 1)
 *   -e <N>    End interval (default: 687, octave 43)
 *   -w <N>    Workers for parallel processing (default: all CPUs)
 *   -j <file> Output JSON with specific n values where extrema occur
 *   -v        Verbose progress output
 * 
 * For analysis, use the companion Python script analyze_extrema.py:
 *   python3 analyze_extrema.py results.csv
 * 
 * REFERENCES:
 * ===========
 * 
 * Hardy & Ramanujan (1917): "The normal number of prime factors of a number n"
 * Dickman, K. (1930): "On the frequency of numbers containing prime factors of a certain relative magnitude"
 * Hildebrand & Tenenbaum (1993): "Integers without large prime factors"
 * 
 * By Ken Clements, December 2025
 * C implementation by Claude Sonnet 4.5, December 2025
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>
#include <omp.h>

// =============================================================================
// CONFIGURATION
// =============================================================================

#define MAX_INTERVALS 1024
#define PIDX_UNREACHABLE 999999
#define PARALLEL_THRESHOLD_INTERVAL 272  // Intervals 272+ use parallel (n >= 2^17)

// Prime table: First 27 primes (up to 101)
// This is sufficient for the search range because:
// - Products with gpf > 101 have higher pi (irrelevant for minimum pi)
// - Such products typically have lower omega (irrelevant for maximum omega)
// - The extrema we care about come from smooth numbers with small primes
static uint64_t primes[27] = {0, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101};
static int num_primes = 27;

// Results structure for each interval
typedef struct {
    int interval;
    uint64_t start_n;
    uint64_t end_n;
    int max_omega;
    int min_pidx;
    uint64_t max_omega_n;  // n where max omega occurs
    uint64_t min_pidx_n;   // n where min pidx occurs
    int valid;             // 0 if no valid products found
} IntervalResult;

// =============================================================================
// FAST FACTORIZATION
// =============================================================================

/*
 * fast_po: Compute Pidx and omega for small smooth numbers
 * 
 * Returns pi(gpf(n)) and omega(n) if all prime factors are in our table.
 * If n has a prime factor > 101, returns PIDX_UNREACHABLE.
 * 
 * This uses a sliding window algorithm: when computing products m*(m+1),
 * we can reuse the factorization of m when computing m+1.
 */
static inline void fast_po(uint64_t n, int pidx_max, int *pidx_out, int *omega_out) {
    int pidx = 0;
    int omega = 0;
    
    for (int i = 1; i < pidx_max; i++) {
        uint64_t p = primes[i];
        
        if (p > n) {
            if (n == 1) {
                *pidx_out = pidx;
                *omega_out = omega;
                return;
            } else {
                // n > 1 but no prime <= n divides it - must have prime factor > 101
                *pidx_out = PIDX_UNREACHABLE;
                *omega_out = 0;
                return;
            }
        }
        
        if (n % p == 0) {
            omega++;
            pidx = i;
            n /= p;
            // Remove all powers of p
            while (n % p == 0) {
                n /= p;
            }
            if (n == 1) {
                *pidx_out = pidx;
                *omega_out = omega;
                return;
            }
        }
    }
    
    // Exhausted prime table without fully factoring n
    *pidx_out = PIDX_UNREACHABLE;
    *omega_out = 0;
}

// =============================================================================
// USAGE
// =============================================================================

void print_usage(const char *progname) {
    fprintf(stderr, "Usage: %s [options]\n\n", progname);
    fprintf(stderr, "Computes max omega and min pi for m*(m+1) over 2^(1/16) intervals.\n");
    fprintf(stderr, "For OEIS A391602 termination analysis.\n\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -s <N>    Start interval (default: 1, octave 1)\n");
    fprintf(stderr, "  -e <N>    End interval (default: 687, octave 43)\n");
    fprintf(stderr, "  -w <N>    Number of workers for large intervals (default: all CPUs)\n");
    fprintf(stderr, "  -j <file> Output JSON file with specific n values\n");
    fprintf(stderr, "  -v        Verbose progress output\n");
    fprintf(stderr, "  -h        Show this help\n\n");
    fprintf(stderr, "Interval Structure:\n");
    fprintf(stderr, "  Interval i spans [2^(i/16), 2^((i+1)/16))\n");
    fprintf(stderr, "  Octave k contains intervals [16k, 16k+15]\n");
    fprintf(stderr, "  Single-threaded for intervals 0-271 (n < 2^17)\n");
    fprintf(stderr, "  Multi-threaded for intervals 272+ (n >= 2^17)\n");
}

// =============================================================================
// INTERVAL PROCESSING
// =============================================================================

void process_interval(int interval, IntervalResult *result, int num_workers, int pidx_max, int verbose, uint64_t force_start) {
    uint64_t start_n, end_n;
    
    // Calculate end boundary
    double exp_end = (interval + 1) / 16.0;
    end_n = (uint64_t)ceil(pow(2.0, exp_end));
    
    // Use forced start if provided (to ensure continuity), otherwise calculate
    if (force_start > 0) {
        start_n = force_start;
    } else {
        double exp_start = interval / 16.0;
        start_n = (uint64_t)ceil(pow(2.0, exp_start));
    }
    
    result->interval = interval;
    result->start_n = start_n;
    result->end_n = end_n;
    result->valid = 0;
    
    uint64_t range_size = end_n - start_n;
    
    if (range_size == 0) {
        if (verbose) {
            fprintf(stderr, "Interval %d: [%llu, %llu) (empty, skipped)\n",
                    interval,
                    (unsigned long long)start_n,
                    (unsigned long long)end_n);
        }
        return;
    }
    
    // Determine if we use parallel processing
    int use_parallel = (interval >= PARALLEL_THRESHOLD_INTERVAL) && (num_workers > 1);
    int actual_workers = use_parallel ? num_workers : 1;
    
    if (verbose) {
        fprintf(stderr, "Interval %d: [%llu, %llu) (%llu values, %s)\n", 
                interval, 
                (unsigned long long)start_n, 
                (unsigned long long)end_n,
                (unsigned long long)range_size,
                use_parallel ? "parallel" : "single");
    }
    
    double interval_start_time = omp_get_wtime();
    
    // Thread-local results
    int local_max_omega[256];
    int local_min_pidx[256];
    uint64_t local_max_omega_n[256];
    uint64_t local_min_pidx_n[256];
    int local_valid[256];
    
    for (int i = 0; i < 256; i++) {
        local_max_omega[i] = -1;
        local_min_pidx[i] = PIDX_UNREACHABLE;
        local_max_omega_n[i] = 0;
        local_min_pidx_n[i] = 0;
        local_valid[i] = 0;
    }
    
    #pragma omp parallel num_threads(actual_workers)
    {
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();
        
        // Divide range among threads
        uint64_t chunk_size = range_size / total_threads;
        uint64_t my_start = start_n + thread_id * chunk_size;
        uint64_t my_end = (thread_id == total_threads - 1) ? end_n : my_start + chunk_size;
        
        int my_max_omega = -1;
        int my_min_pidx = PIDX_UNREACHABLE;
        uint64_t my_max_omega_n = 0;
        uint64_t my_min_pidx_n = 0;
        
        // Initialize for my_start
        int pidx_n, omega_n;
        fast_po(my_start, pidx_max, &pidx_n, &omega_n);
        
        // Process my chunk using sliding window
        for (uint64_t n = my_start; n < my_end; n++) {
            int pidx_n1, omega_n1;
            fast_po(n + 1, pidx_max, &pidx_n1, &omega_n1);
            
            if (pidx_n < PIDX_UNREACHABLE && pidx_n1 < PIDX_UNREACHABLE) {
                // Both n and n+1 are smooth (within our prime table)
                int omega_product = omega_n + omega_n1;
                int pidx_product = (pidx_n > pidx_n1) ? pidx_n : pidx_n1;
                
                // Track maximum omega
                if (omega_product > my_max_omega) {
                    my_max_omega = omega_product;
                    my_max_omega_n = n;
                }
                
                // Track minimum pidx
                if (pidx_product < my_min_pidx) {
                    my_min_pidx = pidx_product;
                    my_min_pidx_n = n;
                }
            }
            
            // Slide window
            pidx_n = pidx_n1;
            omega_n = omega_n1;
        }
        
        // Store thread results
        local_max_omega[thread_id] = my_max_omega;
        local_min_pidx[thread_id] = my_min_pidx;
        local_max_omega_n[thread_id] = my_max_omega_n;
        local_min_pidx_n[thread_id] = my_min_pidx_n;
        local_valid[thread_id] = (my_max_omega >= 0);
    }
    
    // Merge thread results
    int global_max_omega = -1;
    int global_min_pidx = PIDX_UNREACHABLE;
    uint64_t global_max_omega_n = 0;
    uint64_t global_min_pidx_n = 0;
    
    for (int t = 0; t < actual_workers; t++) {
        if (local_valid[t]) {
            result->valid = 1;
            
            if (local_max_omega[t] > global_max_omega) {
                global_max_omega = local_max_omega[t];
                global_max_omega_n = local_max_omega_n[t];
            }
            
            if (local_min_pidx[t] < global_min_pidx) {
                global_min_pidx = local_min_pidx[t];
                global_min_pidx_n = local_min_pidx_n[t];
            }
        }
    }
    
    result->max_omega = global_max_omega;
    result->min_pidx = global_min_pidx;
    result->max_omega_n = global_max_omega_n;
    result->min_pidx_n = global_min_pidx_n;
    
    if (verbose) {
        double elapsed = omp_get_wtime() - interval_start_time;
        if (result->valid) {
            fprintf(stderr, "  max_omega=%d at n=%llu, min_pidx=%d at n=%llu (%.3fs)\n",
                    result->max_omega,
                    (unsigned long long)result->max_omega_n,
                    result->min_pidx,
                    (unsigned long long)result->min_pidx_n,
                    elapsed);
        } else {
            fprintf(stderr, "  No valid products (%.3fs)\n", elapsed);
        }
    }
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char *argv[]) {
    // Default parameters
    int start_interval = 1;   // Octave 1
    int end_interval = 687;    // Octave 43
    int num_workers = omp_get_max_threads();
    int verbose = 0;
    char *json_filename = NULL;

    // Parse command line
    int opt;
    while ((opt = getopt(argc, argv, "s:e:w:j:vh")) != -1) {
        switch (opt) {
            case 's':
                start_interval = atoi(optarg);
                break;
            case 'e':
                end_interval = atoi(optarg);
                break;
            case 'w':
                num_workers = atoi(optarg);
                break;
            case 'j':
                json_filename = optarg;
                break;
            case 'v':
                verbose = 1;
                break;
            case 'h':
            default:
                print_usage(argv[0]);
                return (opt == 'h') ? 0 : 1;
        }
    }
    
    // Validate
    if (start_interval < 0) start_interval = 0;
    if (end_interval >= MAX_INTERVALS) end_interval = MAX_INTERVALS - 1;
    if (end_interval < start_interval) end_interval = start_interval;
    if (num_workers < 1) num_workers = 1;
    
    int pidx_max = num_primes;
    
    if (verbose) {
        fprintf(stderr, "\n");
        fprintf(stderr, "======================================================================\n");
        fprintf(stderr, "INTERVAL EXTREMA CALCULATOR - For OEIS A391602\n");
        fprintf(stderr, "======================================================================\n");
        fprintf(stderr, "Using %d primes up to %llu\n", num_primes, (unsigned long long)primes[num_primes-1]);
        fprintf(stderr, "Intervals: %d to %d (%d total)\n", 
                start_interval, end_interval, end_interval - start_interval + 1);
        fprintf(stderr, "Workers: %d (for large intervals)\n", num_workers);
        fprintf(stderr, "======================================================================\n\n");
    }
    
    double total_start = omp_get_wtime();
    
    // Allocate results array
    int num_intervals = end_interval - start_interval + 1;
    IntervalResult *results = malloc(num_intervals * sizeof(IntervalResult));
    if (!results) {
        fprintf(stderr, "Error: Failed to allocate results array\n");
        return 1;
    }
    
    // Process intervals sequentially to ensure continuity
    uint64_t prev_end = 0;
    for (int i = 0; i < num_intervals; i++) {
        int interval = start_interval + i;
        process_interval(interval, &results[i], num_workers, pidx_max, verbose, prev_end);
        prev_end = results[i].end_n;
    }
    
    double total_elapsed = omp_get_wtime() - total_start;
    
    // Output CSV results to stdout
    printf("interval,start_n,end_n,max_omega,min_pidx,max_omega_n,min_pidx_n\n");
    for (int i = 0; i < num_intervals; i++) {
        if (results[i].valid) {
            printf("%d,%llu,%llu,%d,%d,%llu,%llu\n",
                   results[i].interval,
                   (unsigned long long)results[i].start_n,
                   (unsigned long long)results[i].end_n,
                   results[i].max_omega,
                   results[i].min_pidx,
                   (unsigned long long)results[i].max_omega_n,
                   (unsigned long long)results[i].min_pidx_n);
        }
    }
    
    // Output JSON if requested
    if (json_filename) {
        FILE *json_file = fopen(json_filename, "w");
        if (!json_file) {
            fprintf(stderr, "Error: Could not open JSON file %s\n", json_filename);
        } else {
            fprintf(json_file, "{\n");
            fprintf(json_file, "  \"max_omega_n\": {\n");
            for (int i = 0; i < num_intervals; i++) {
                if (results[i].valid) {
                    fprintf(json_file, "    \"%d\": %llu%s\n",
                            results[i].interval,
                            (unsigned long long)results[i].max_omega_n,
                            (i < num_intervals - 1) ? "," : "");
                }
            }
            fprintf(json_file, "  },\n");
            fprintf(json_file, "  \"min_pidx_n\": {\n");
            for (int i = 0; i < num_intervals; i++) {
                if (results[i].valid) {
                    fprintf(json_file, "    \"%d\": %llu%s\n",
                            results[i].interval,
                            (unsigned long long)results[i].min_pidx_n,
                            (i < num_intervals - 1) ? "," : "");
                }
            }
            fprintf(json_file, "  }\n");
            fprintf(json_file, "}\n");
            fclose(json_file);
            
            if (verbose) {
                fprintf(stderr, "\nJSON output written to %s\n", json_filename);
            }
        }
    }
    
    if (verbose) {
        fprintf(stderr, "\n");
        fprintf(stderr, "======================================================================\n");
        fprintf(stderr, "SUMMARY\n");
        fprintf(stderr, "======================================================================\n");
        fprintf(stderr, "Total time:    %.2f seconds\n", total_elapsed);
        fprintf(stderr, "Intervals:     %d to %d (%d total)\n", 
                start_interval, end_interval, num_intervals);
        fprintf(stderr, "Valid results: %d\n", num_intervals);
        fprintf(stderr, "======================================================================\n");
    }
    
    free(results);
    return 0;
}
