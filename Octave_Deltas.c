/*
 * Octave_Deltas.c
 * 
 * Counts delta values (Pidx - omega) for n(n+1) products, organized by octave.
 * Uses OpenMP for parallelization.
 *
 * Compile:
 *   Linux/Pi:  gcc -O3 -march=native -fopenmp -o octave_deltas Octave_Deltas.c -lm
 *   Mac:       clang -O3 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include \
 *              -L/opt/homebrew/opt/libomp/lib -lomp -o octave_deltas Octave_Deltas.c -lm
 *
 * Usage:
 *   ./octave_deltas [options]
 *
 * Options:
 *   -s <N>    Start octave (default: 1)
 *   -e <N>    End octave (default: 43)
 *   -w <N>    Number of workers (default: all CPUs)
 *   -c        Output as CSV
 *   -x        Extended output lists all n with delta in range
 *   -h        Show help
 *
 * Examples:
 *   ./octave_deltas -s 30 -e 40 -w 8
 *   ./octave_deltas -s 35 -e 45 -c > results.csv
 *
 * Example Output:
 * 
 * 
 *Octave |     δ=0 |     δ=1 |     δ=2 |     δ=3 |     δ=4 |
 *-------+---------+---------+---------+---------+---------+
 *     1 |       2 |       0 |       0 |       0 |       0 |
 *     2 |       1 |       2 |       1 |       0 |       0 |
 *     3 |       4 |       0 |       2 |       2 |       0 |
 *     4 |       2 |       2 |       0 |       2 |       1 |
 *     5 |       1 |       6 |       2 |       4 |       2 |
 *     6 |       2 |       7 |       2 |       1 |       5 |
 *     7 |       1 |       2 |       6 |       7 |       3 |
 *     8 |       2 |       2 |       8 |       3 |      10 |
 *     9 |       2 |       4 |       3 |       3 |       4 |
 *    10 |       1 |       2 |       3 |      10 |       9 |
 *    11 |       3 |       3 |       6 |       9 |       6 |
 *    12 |       1 |       3 |       7 |       8 |       7 |
 *    13 |       2 |       4 |       5 |      10 |       9 |
 *    14 |       0 |       3 |       9 |       5 |      14 |
 *    15 |       0 |       3 |       1 |       5 |       7 |
 *    16 |       1 |       1 |       4 |       6 |      13 |
 *    17 |       1 |       1 |       7 |       7 |       8 |
 *    18 |       0 |       1 |       5 |       7 |       7 |
 *    19 |       1 |       3 |       2 |       2 |       8 |
 *    20 |       0 |       0 |       5 |       8 |      10 |
 *    21 |       0 |       2 |       3 |       6 |       7 |
 *    22 |       0 |       2 |       2 |       2 |       4 |
 *    23 |       0 |       3 |       0 |       2 |       7 |
 *    24 |       0 |       1 |       1 |       3 |       4 |
 *    25 |       0 |       0 |       0 |       4 |       4 |
 *    26 |       0 |       1 |       1 |       4 |       6 |
 *    27 |       0 |       0 |       0 |       3 |       2 |
 *    28 |       0 |       0 |       0 |       3 |       6 |
 *    29 |       0 |       0 |       1 |       0 |       5 |
 *    30 |       0 |       0 |       2 |       0 |       1 |
 *    31 |       0 |       0 |       0 |       2 |       1 |
 *    32 |       0 |       0 |       0 |       0 |       1 |
 *    33 |       0 |       0 |       0 |       0 |       0 |
 *    34 |       0 |       0 |       0 |       1 |       1 |
 *    35 |       0 |       0 |       1 |       0 |       2 |
 *    36 |       0 |       0 |       0 |       0 |       0 |
 *    37 |       0 |       0 |       0 |       0 |       0 |
 *    38 |       0 |       0 |       0 |       2 |       0 |
 *    39 |       0 |       0 |       0 |       0 |       1 |
 *    40 |       0 |       0 |       1 |       1 |       1 |
 *    41 |       0 |       0 |       0 |       0 |       0 |
 *    42 |       0 |       0 |       0 |       0 |       0 |
 *    43 |       0 |       0 |       0 |       0 |       0 |
 *    44 |       0 |       0 |       0 |       0 |       1 |
 * 
 * 
 * By Ken Clements, December 2025
 * C version by Claude Sonnet 4.5, December 2025
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


#define MAX_OCTAVES 64
#define MAX_DELTA 10
#define PIDX_UNREACHABLE 999999

// Global prime table
static uint64_t primes[27] = {0, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101};
static int num_primes = 27;


// =============================================================================
// FAST FACTORIZATION
// =============================================================================

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
                *pidx_out = PIDX_UNREACHABLE;
                *omega_out = 0;
                return;
            }
        }
        
        if (n % p == 0) {
            omega++;
            pidx = i;
            n /= p;
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
    
    *pidx_out = PIDX_UNREACHABLE;
    *omega_out = 0;
}

// =============================================================================
// USAGE
// =============================================================================

void print_usage(const char *progname) {
    fprintf(stderr, "Usage: %s [options]\n\n", progname);
    fprintf(stderr, "Counts delta values (Pidx - omega) for n(n+1) products by octave.\n\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -s <N>    Start octave (default: 3)\n");
    fprintf(stderr, "  -e <N>    End octave (default: 43, max: %d)\n", MAX_OCTAVES - 1);
    fprintf(stderr, "  -w <N>    Number of workers (default: all CPUs)\n");
    fprintf(stderr, "  -c        Output as CSV (for piping to file)\n");
    fprintf(stderr, "  -x        Output every found delta (for piping to file)\n");
    fprintf(stderr, "  -h        Show this help\n\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  %s -s 30 -e 40 -w 8\n", progname);
    fprintf(stderr, "  %s -s 35 -e 45 -c > results.csv\n", progname);
    fprintf(stderr, "\nOctave N contains n values from 2^N to 2^(N+1) - 1\n");
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char *argv[]) {
    // Default parameters
    int delta_limit = 5;
    int start_octave = 1;
    int end_octave = 43;
    int num_workers = omp_get_max_threads();
    int csv_output = 0;
    int extended_output = 0;

    fprintf(stderr, "Using %d primes up to %llu \n", num_primes, primes[num_primes-1]);

    // Parse command line
    int opt;
    while ((opt = getopt(argc, argv, "s:e:w:c:xh")) != -1) {
        switch (opt) {
            case 's':
                start_octave = atoi(optarg);
                break;
            case 'e':
                end_octave = atoi(optarg);
                if (end_octave >= MAX_OCTAVES) end_octave = MAX_OCTAVES - 1;
                break;
            case 'w':
                num_workers = atoi(optarg);
                break;
            case 'c':
                csv_output = 1;
                break;
            case 'x':
                extended_output = 1;
                break;
            case 'h':
            default:
                print_usage(argv[0]);
                return (opt == 'h') ? 0 : 1;
        }
    }
    
    // Validate
    if (start_octave < 1) start_octave = 1;
    if (end_octave < start_octave) end_octave = start_octave;
    if (num_workers < 1) num_workers = 1;
    
    int pidx_max = num_primes;
    
    // Allocate counts array: counts[octave][delta]
    int counts[MAX_OCTAVES][MAX_DELTA];
    memset(counts, 0, sizeof(counts));
    
    // Print header (unless CSV)
    if (!csv_output) {
        fprintf(stderr, "\nCounting Pidx - omega values of n(n+1) in octave ranges of n\n");
        fprintf(stderr, "Octaves: %d to %d\n", 
                start_octave, end_octave);
        fprintf(stderr, "Delta limit: %d\n", delta_limit);
        fprintf(stderr, "Workers: %d\n\n", num_workers);
    }
    
    double total_start = omp_get_wtime();
    
    // Process each octave
    for (int octave = start_octave; octave <= end_octave; octave++) {
        uint64_t oct_start = 1ULL << octave;
        uint64_t oct_end = 1ULL << (octave + 1);
        uint64_t oct_size = oct_end - oct_start;
        
        if (!csv_output) {
            fprintf(stderr, "Processing octave %d (%llu values)\n", 
                    octave, (unsigned long long)oct_size);
        }
        
        double oct_start_time = omp_get_wtime();
        
        // Parallel processing of this octave
        #pragma omp parallel num_threads(num_workers)
        {
            // Local counts for this thread
            uint64_t local_counts[MAX_DELTA];
            memset(local_counts, 0, sizeof(local_counts));
            
            int thread_id = omp_get_thread_num();
            int total_threads = omp_get_num_threads();
            
            // Divide octave range among threads
            uint64_t chunk_size = oct_size / total_threads;
            uint64_t my_start = oct_start + thread_id * chunk_size;
            uint64_t my_end = (thread_id == total_threads - 1) ? oct_end : my_start + chunk_size;
            
            // Initialize for my_start
            int pidx_n, omega_n;
            fast_po(my_start, pidx_max, &pidx_n, &omega_n);
            
            // Process my chunk
            for (uint64_t n = my_start; n < my_end; n++) {
                int pidx_n1, omega_n1;
                fast_po(n + 1, pidx_max, &pidx_n1, &omega_n1);
                
                if (pidx_n < PIDX_UNREACHABLE && pidx_n1 < PIDX_UNREACHABLE) {
                    int omega_product = omega_n + omega_n1;
                    int pidx_product = (pidx_n > pidx_n1) ? pidx_n : pidx_n1;
                    int delta = pidx_product - omega_product;
                    
                    if (delta >= 0 && delta < delta_limit) {
                        local_counts[delta]++;
                        if (extended_output) {
                            printf("n_delta[%d].append(%llu)\n", delta, (unsigned long long)n );
                        }
                    }
                }
                
                // Slide window
                pidx_n = pidx_n1;
                omega_n = omega_n1;
            }
            
            // Merge into global counts
            #pragma omp critical
            {
                for (int d = 0; d < delta_limit; d++) {
                    counts[octave][d] += local_counts[d];
                }
            }
        }
        
        double oct_elapsed = omp_get_wtime() - oct_start_time;
        
        if (!csv_output && oct_elapsed > 0.1) {
            double rate = oct_size / oct_elapsed / 1e6;
            fprintf(stderr, "Octave %d done in %.2fs (%.1fM/s)\n", 
                    octave, oct_elapsed, rate);
        }
    }
    
    double total_elapsed = omp_get_wtime() - total_start;
    
    // Output results
    if (csv_output) {
        // CSV header
        printf("octave");
        for (int d = 0; d < delta_limit; d++) {
            printf(",delta_%d", d);
        }
        printf("\n");
        
        // CSV data
        for (int oct = start_octave; oct <= end_octave; oct++) {
            printf("%d", oct);
            for (int d = 0; d < delta_limit; d++) {
                printf(",%d", counts[oct][d]);
            }
            printf("\n");
        }
    } else {
        // Pretty-printed table
        printf("\n");
        printf("======================================================================\n");
        printf("FINAL RESULTS\n");
        printf("======================================================================\n\n");
        
        // Header row
        printf("%6s |", "Octave");
        for (int d = 0; d < delta_limit; d++) {
            printf("  %6s%d |", "δ=", d);
        }
        printf("\n");
        
        // Separator
        printf("-------+");
        for (int d = 0; d < delta_limit; d++) {
            printf("---------+");
        }
        printf("\n");
        
        // Data rows
        for (int oct = start_octave; oct <= end_octave; oct++) {
            printf("%6d |", oct);
            for (int d = 0; d < delta_limit; d++) {
                printf(" %7d |", counts[oct][d]);
            }
            printf("\n");
        }
        
        printf("\n======================================================================\n");
        printf("SUMMARY\n");
        printf("======================================================================\n");
        printf("Total time:    %.2f seconds\n", total_elapsed);
        printf("Octaves:       %d to %d (%d total)\n", 
               start_octave, end_octave, end_octave - start_octave + 1);
        printf("======================================================================\n");
    }
    
    return 0;
}
