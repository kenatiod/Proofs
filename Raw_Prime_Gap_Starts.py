# Raw_Prime_Gap_Starts.py
"""
This program examines the factorizations of the products of
strings of integers at the start of prime gaps. In this version
we just add up the raw omega counts of the integers in the 
strings.

By Ken Clements, November 11, 2025.
"""

from sympy import primefactors, primerange
PRINT_DETAILS = False
START_PRIME_INDEX = 200
MAX_PRIME_RANGE = 50_000

print(f"Searching for strings of consecutive integers at the start of prime gaps, for primes up to {MAX_PRIME_RANGE:,}:")

primes = [0] + list(primerange(2, MAX_PRIME_RANGE))
for s_length in range(4, 20):
    max_omega = 1
    min_high_prime = 200_000
    for i in range(START_PRIME_INDEX, len(primes)-1):
        p_start = primes[i]
        p_end = primes[i+1]
        if p_end - p_start <= s_length: continue
        #product = p_start + 1
        integer_prime_factors = primefactors(p_start + 1)
        raw_omega_sum = len(integer_prime_factors)
        max_factor = integer_prime_factors[-1]
        for k in range(s_length - 1):
            #product *= p_start + 2 + k
            integer_prime_factors = primefactors(p_start + 2 + k)
            raw_omega_sum += len(integer_prime_factors)
            max_factor = max(max_factor, integer_prime_factors[-1])
        #pf = primefactors(product)
        #if PRINT_DETAILS and s_length < 7 and p_start+1 < 1000:
        #    print(f"{p_start+1:5}, {pf}")
        #if len(pf) > max_omega: max_omega = len(pf)
        max_omega = max(max_omega, raw_omega_sum)
        min_high_prime = min(min_high_prime, max_factor)

    print(f"Tuple length = {s_length:2}, {max_omega=:2} with prime {primes[max_omega]:3} and {min_high_prime=:3} Delta is {min_high_prime - primes[max_omega]:3}")
print("End of Program")    

