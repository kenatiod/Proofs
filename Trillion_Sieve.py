# Pi-Complete_Sieve_13_tupel_V2.py

"""
This program searches for pi-complete numbers (numbers having a complete set of
prime factors down to 2) that are the product of strings of consecutive integers, by means
of a prime number sieve. It builds lists of prime numbers, and for each integer between
it calculates omega(n), the number of prime factors of that integer, and the index
(starts with 1 for the prime 2) of the greatest prime factor, max_prime_idx. 
Based on these values, a test can be made if a product of a string of these
integers is pi-complete by adding their omega values,
subtracting the number of shared prime factors, and testing
that tally against the greatest index of their prime factors. If
those are equal, the product is pi-complete. The reason we care about this
is because when searching for solutions to Diophantine equations that are
factorials, all factorials must be pi-complete. If no possible solutions
are pi-complete beyond a certain bound, then only numbers up to that bound
need be tested as possible solutions.

In this version of the program, the operation is carried out in segments. Each
segment fills a fixed length, unsigned 8 bit array. The number, n, is
represented by the array index plus the array start value (initialized to 0).
Each array is sieved to find sequences of integers between prime numbers,
(i.e. in prime gaps), and those are tested to see if their multiplicative products
form pi-complete numbers. In this version, only strings at the start of
prime gaps are tested (only those which could be solutions for A! = C!/B!).

In order to save memory, omega(n) and the max_prime_idx(n), for each array location,
are limited to 4 bits to reduce memory (combined into 8 bits for easy access). 
This relies on the sum of omega(n) always landing within the 4 bits for any possible
pi-complete product. The max prime index limits out at 15, (prime(14) = 47) which 
is well beyond any prime factor that could be in a pi-complete product for any
practical search range. For example, primorial numbers are the smallest
pi-complete numbers with that count of unique prime factors, and just 15
of those gives the primorial number 614,889,782,588,491,410.

The overall program has a goal limit and keeps sieving segments until a prime
is reached above that goal. The result lists are joined along the way. Newly found
prime numbers stop being stored when they are half way to the goal, as those will
not cause sieve hits in any subsequent segments.

SSD PAGING VERSION:
This version writes prime data to disk files, one per segment, allowing the program
to handle arbitrarily large goals without RAM limitations. Each segment file contains
the delta-encoded primes discovered in that segment. The program can be stopped and
restarted, automatically resuming from the last completed segment. When sieving a new
segment, the program streams through all previous segment files to perform the pre-sieving
operation, keeping memory usage constant regardless of goal size.

Examples of prime factor counts and maximum prime index values for starting numbers:
n:               2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
omega(n):        1,  1,  1,  1,  2,  1,  1,  1,  2,  1,  2,  1,  2,  2,  1,  1,  2,  1,  2,  
max_prime(n)     2,  3,  2,  5,  3,  7,  2,  3,  5, 11,  3, 13,  7,  5,  2, 17,  3, 19,  5,  
max_prime_idx(n) 1,  2,  1,  3,  2,  4,  1,  2,  3,  5,  2,  6,  4,  3,  1,  7,  2,  8,  3,  

n:              21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39
omega(n):        2,  2,  1,  2,  1,  2,  1,  2,  1,  3,  1,  1,  2,  2,  2,  2,  1,  2,  2
max_prime(n)     7, 11, 23,  3,  5, 13,  3,  7, 29,  5, 31,  2, 11, 17,  7,  3, 37, 19, 13
max_prime_idx(n) 4,  5,  9,  2,  3,  6,  2,  4, 10,  3, 11,  1,  5,  7,  4,  2, 12,  8,  6

When the program finds a new prime number, it saves the delta/2 from the last found prime
number LEB128 style encoded in an 8 bit unsigned array. These deltas are written to the
current segment file as they are discovered.

Run as: python Pi-Complete_Sieve_SSD.py [goal] [segment_size] [data_directory]
or omit the parameters to take the defaults.

By Ken Clements, November 6, 2025
"""

import sys
import array
import os
import struct
from math import log
from sympy import prime, primefactors, factorint

# Default parameters:
goal = 1_000_000_000_000  # Default to 1 trillion
segment_size = 10_000_000_000  # 10 billion integers per segment
data_directory = "primes_data"  # Directory to store segment files

if len(sys.argv) > 1:
    goal = int(sys.argv[1])
if len(sys.argv) > 2:
    segment_size = int(sys.argv[2])
if len(sys.argv) > 3:
    data_directory = sys.argv[3]
if segment_size % 2 == 1:  # Buffer size has to be even
    segment_size += 1

MAX_IDX = 0x0F  # cap for 4-bit storage

MAX_TUPLE = 13   # Test products of strings of up to this many integers.
# Define lookup tables for shared factors (computed once) 
# These are indexed by [number of integers -2][start integer (mod divisor_period)]

divisor_period, i = 1, 1
while prime(i) < MAX_TUPLE:     # Calculate divisor period for this many prime divisors
   divisor_period *= prime(i)   # i.e. the divisor period for products that fit the 
   i += 1                       # first three primes is 2*3*5 = 30.

correction_matrix = []          # Build a correction matrix out of correction vectors
for m in range(1, MAX_TUPLE + 1):
    correction_vector = []
    for n in range(divisor_period):
        product = divisor_period + n
        pfl_sum = len(primefactors(divisor_period + n))
        for j in range(m):
            product *= (divisor_period+n+j+1) 
            pfl_sum += len(primefactors(divisor_period+n+j+1))
        pfl_product = len(primefactors(product)) 
        correction_vector.append(pfl_sum - pfl_product)
    correction_matrix.append(correction_vector)

def is_pi_complete(n):
    if n < 2: return False
    pf = primefactors(n)
    lpf = len(pf)
    return prime(lpf) == pf[-1]

def has_FEF(n):
    if not is_pi_complete(n): return False
    fi = factorint(n)
    exponents = list(fi.values())
    lxp = len(exponents)
    if exponents[-1] != 1: return False
    if lxp == 1: return True
    exp = 1
    for i in range(2, lxp+1):
        if exponents[-i] < exp: return False
        exp = exponents[-i]
    return True



class SegmentFile:
    """
    Manages reading and writing of segment files containing delta-encoded primes.
    
    File format:
    - Header (64 bytes fixed):
        - Magic number (4 bytes): 'PSEG'
        - Version (4 bytes): 1
        - Segment number (8 bytes)
        - Segment size (8 bytes)
        - Segment start (8 bytes)
        - Segment end / last_prime_stored (8 bytes)
        - Prime count at end of segment (8 bytes)
        - Reserved (16 bytes)
    - Data section (variable):
        - LEB128-encoded delta/2 values for primes in this segment
    """
    
    HEADER_SIZE = 64
    MAGIC = b'PSEG'
    VERSION = 1
    
    def __init__(self, directory, segment_size_billions, segment_number, mode='w'):
        """
        Initialize segment file for reading or writing.
        
        Args:
            directory: Directory to store segment files
            segment_size_billions: Size of segments in billions (for filename)
            segment_number: Sequential segment number
            mode: 'w' for writing new segment, 'r' for reading existing segment
        """
        self.directory = directory
    
        self.segment_size_billions = segment_size_billions
        self.segment_number = segment_number
        self.mode = mode
        
        # Generate filenames: use .tmp while writing, rename to .dat when complete
        self.filename_base = f"primes_{segment_size_billions}B_seg{segment_number:04d}"
        self.filename = os.path.join(directory, f"{self.filename_base}.dat")
        self.filename_tmp = os.path.join(directory, f"{self.filename_base}.tmp")
        
        self.file_handle = None
        self.write_buffer = array.array('B')  # Buffer for writing delta values
        self.buffer_size = 1024 * 1024  # 1 MB write buffer
        
        # Metadata (set when closing write mode or reading header)
        self.metadata = {
            'segment_number': segment_number,
            'segment_size': 0,
            'segment_start': 0,
            'segment_end': 0,
            'prime_count': 0
        }

    
    def __enter__(self):
        """Context manager entry."""
        if self.mode == 'w':
            # Clean up any existing .tmp file from previous incomplete run
            if os.path.exists(self.filename_tmp):
                print(f"    Removing incomplete temp file: {self.filename_tmp}")
                os.remove(self.filename_tmp)
            
            # Write to .tmp file
            self.file_handle = open(self.filename_tmp, 'wb')
            # Write placeholder header (will update on close)
            self.file_handle.write(b'\x00' * self.HEADER_SIZE)
        elif self.mode == 'r':
            self.file_handle = open(self.filename, 'rb')
            self._read_header()
        return self

    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.mode == 'w' and self.file_handle:
            self._flush_write_buffer()
            self._write_header()
            self.file_handle.close()
            self.file_handle = None
            
            # Only rename to .dat if no exception occurred
            if exc_type is None:
                os.rename(self.filename_tmp, self.filename)
                print(f"    Completed and saved: {self.filename}")
            else:
                print(f"    Error occurred, temp file not renamed: {self.filename_tmp}")
        elif self.file_handle:
            self.file_handle.close()
        return False

    
    def _write_header(self):
        """Write metadata header at the beginning of the file."""
        self.file_handle.seek(0)
        header = struct.pack(
            '<4sI5Q16x',  # < = little-endian, 4s = 4 char string, I = uint32, Q = uint64, 16x = 16 padding bytes
            self.MAGIC,
            self.VERSION,
            self.metadata['segment_number'],
            self.metadata['segment_size'],
            self.metadata['segment_start'],
            self.metadata['segment_end'],
            self.metadata['prime_count']
        )
        self.file_handle.write(header)
    
    def _read_header(self):
        """Read metadata header from file."""
        self.file_handle.seek(0)
        header_data = self.file_handle.read(self.HEADER_SIZE)
        
        magic, version, seg_num, seg_size, seg_start, seg_end, prime_count = struct.unpack(
            '<4sI5Q16x', header_data
        )
        
        if magic != self.MAGIC:
            raise ValueError(f"Invalid segment file magic number in {self.filename}")
        if version != self.VERSION:
            raise ValueError(f"Unsupported segment file version {version} in {self.filename}")
        
        self.metadata = {
            'segment_number': seg_num,
            'segment_size': seg_size,
            'segment_start': seg_start,
            'segment_end': seg_end,
            'prime_count': prime_count
        }
    
    def append_delta(self, delta):
        """
        Append a delta-encoded prime (LEB128 format) to the segment file.
        
        Args:
            delta: The delta/2 value to encode and store
        """
        # LEB128 encoding: 7 bits per byte, MSB indicates continuation
        while delta > 127:
            self.write_buffer.append((delta & 0x7F) | 0x80)
            delta >>= 7
        self.write_buffer.append(delta)
        
        # Flush buffer if it's getting large
        if len(self.write_buffer) >= self.buffer_size:
            self._flush_write_buffer()
    
    def _flush_write_buffer(self):
        """Write buffered delta values to disk."""
        if len(self.write_buffer) > 0 and self.file_handle:
            self.file_handle.write(self.write_buffer.tobytes())
            self.write_buffer = array.array('B')
    
    def set_metadata(self, segment_size, segment_start, segment_end, prime_count):
        """
        Set metadata for the segment (called before closing write mode).
        
        Args:
            segment_size: Size of the segment in integers
            segment_start: Starting value of the segment
            segment_end: Ending value (last_prime_stored)
            prime_count: Total number of primes found through this segment
        """
        self.metadata['segment_size'] = segment_size
        self.metadata['segment_start'] = segment_start
        self.metadata['segment_end'] = segment_end
        self.metadata['prime_count'] = prime_count
    
    def iter_primes(self, initial_prime=2, initial_index=1):
        """
        Generator that yields primes from this segment file.
        Reads from current file position, decoding LEB128 delta values.
        Properly handles LEB128 values that span across chunk boundaries.
        
        Args:
            initial_prime: The prime value to start from (for delta decoding)
            initial_index: The prime index to start from
            
        Yields:
            Tuples of (prime_value, prime_index)
        """
        if self.mode != 'r':
            raise ValueError("Can only iterate primes in read mode")
        
        current_prime = initial_prime
        current_index = initial_index
        
        # Read in chunks for efficiency
        chunk_size = 1024 * 1024  # 1 MB chunks
        
        # Buffer for handling LEB128 values that span chunk boundaries
        leftover_bytes = bytearray()
        
        while True:
            chunk = self.file_handle.read(chunk_size)
            if not chunk and not leftover_bytes:
                break
            
            # Prepend any leftover bytes from previous chunk
            if leftover_bytes:
                data = leftover_bytes + chunk
                leftover_bytes = bytearray()
            else:
                data = chunk
            
            i = 0
            while i < len(data):
                # Decode LEB128
                delta = 0
                shift = 0
                start_i = i
                
                while i < len(data):
                    byte = data[i]
                    i += 1
                    delta |= (byte & 0x7F) << shift
                    
                    # Check if this is the last byte of the LEB128 value
                    if not (byte & 0x80):
                        # Complete LEB128 value decoded
                        break
                    
                    shift += 7
                    
                    # Safety check: LEB128 for 64-bit values shouldn't exceed 10 bytes
                    if shift >= 70:
                        raise ValueError(f"LEB128 decode error: value too large at position {i}")
                
                # Check if we hit end of data while still in continuation mode
                if i >= len(data) and (byte & 0x80):
                    # Incomplete LEB128 value - save bytes for next chunk
                    leftover_bytes = bytearray(data[start_i:])
                    break
                
                # We have a complete delta value
                # Convert delta/2 back to actual prime gap
                current_prime += (delta << 1)
                current_index += 1
                
                yield (current_prime, current_index)
            
            # If no more data from file and no leftover bytes, we're done
            if not chunk:
                break

def scan_existing_segments(directory, segment_size):
    """
    Scan the data directory for existing segment files matching the current segment size.
    
    Args:
        directory: Directory containing segment files
        segment_size: Current segment size setting
        
    Returns:
        List of (segment_number, filename) tuples, sorted by segment number
    """
    if not os.path.exists(directory):
        return []
    
    segment_size_billions = segment_size // 1_000_000_000
    prefix = f"primes_{segment_size_billions}B_seg"
    suffix = ".dat"
    
    segments = []
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith(suffix):
            try:
                # Extract segment number from filename
                seg_num_str = filename[len(prefix):-len(suffix)]
                seg_num = int(seg_num_str)
                segments.append((seg_num, os.path.join(directory, filename)))
            except ValueError:
                continue
    
    segments.sort()  # Sort by segment number
    return segments


def load_segment_metadata(filename, segment_size_billions):
    """
    Load metadata from an existing segment file without reading all the data.
    
    Args:
        filename: Path to segment file
        
    Returns:
        Dictionary containing segment metadata
    """
    with SegmentFile(data_directory, segment_size_billions, 0, mode='r') as sf:
        sf.filename = filename  # Override to read specific file
        sf.file_handle = open(filename, 'rb')
        sf._read_header()
        return sf.metadata


def cleanup_temp_files(directory, segment_size_billions):
    """
    Remove any .tmp files left over from previous incomplete runs.
    
    Args:
        directory: Directory containing segment files
        segment_size_billions: Size identifier for filename matching
    """
    if not os.path.exists(directory):
        return
    
    removed_count = 0
    pattern_prefix = f"primes_{segment_size_billions}B_seg"
    
    for filename in os.listdir(directory):
        if filename.startswith(pattern_prefix) and filename.endswith('.tmp'):
            tmp_path = os.path.join(directory, filename)
            try:
                os.remove(tmp_path)
                removed_count += 1
                print(f"  Removed incomplete temp file: {filename}")
            except Exception as e:
                print(f"  Warning: Could not remove {filename}: {e}")
    
    if removed_count > 0:
        print(f"Cleaned up {removed_count} incomplete temp file(s)")
    
    return removed_count



def build_segment(segment_number, segment_size, segment_array, goal, 
                 segment_file, previous_segment_files, 
                 last_prime_stored, prime_count):
    """
    This routine builds a segment of the sieve within the 8 bit unsigned segment_size array
    of memory. After the first segment, this array uses an offset, last_prime_stored+1, to map
    the array index into the numbers being sieved. 

    As the locations in the array are sieved, they accumulate the count of distinct prime factors,
    omega(n), and the index of the greatest of those prime factors. This works because
    each new prime is found only once, so will only hit each composite number once.
    
    Prime numbers are found by the sieving process going through the segment addresses,
    and these are written to the segment file in delta/2 encoding for use on the next segment.

    The algorithm is extremely fast for a single segment because there are only simple
    additions and value tests. However, when the goal cannot be reached in a single segment
    because of memory restrictions, things slow down because all successive segments
    have to be pre-sieved by all the prime numbers found from the start. This requires a
    remaindering operation performed on the segment start value by each prime number
    so as to find the starting offset within the segment to start sieving by that
    prime number. 
    
    In this SSD version, primes from previous segments are streamed from disk files rather
    than held in RAM, keeping memory usage constant.
    
    Args:
        segment_number: Sequential number of this segment
        segment_size: Size of the segment array
        segment_array: Working memory array for sieving
        goal: Overall search goal
        segment_file: SegmentFile object for writing new primes
        previous_segment_files: List of filenames for previous segments to read
        last_prime_stored: The last prime found in previous segments
        prime_count: Count of primes found so far
        
    Returns:
        Tuple of (found_list, last_prime_stored, prime_count)
    """
    segment_size_billions = segment_size // 1_000_000_000
    prime_storing_limit = (goal // 2) + 100  # Store primes up to half the goal plus some margin
    
    # Start at 0 for the first segment, then resume at last_prime_stored+1,
    # with last prime being odd, the resume point will always be even so we
    # can fill the segment buffer with the hits from 2.
    segment_start = 0 if last_prime_stored < 3 else last_prime_stored + 1
    found_list = []
    
    print(f"\nBuilding segment {segment_number} from {segment_start:,} to {segment_start + segment_size:,}")
    
    # Initialize segment buffer: evens get one hit of 2; odds start at 0
    for i in range(0, segment_size, 2):
        segment_array[i] = 0x11  # A single mark by prime 1 (index for 2)
        segment_array[i+1] = 0   # zero for all the odd numbers
    
    # If this segment is not the first, pre-sieve with primes from previous segments.
    if segment_start > 0:
        print(f"  Pre-sieving with primes from {len(previous_segment_files)} previous segment file(s)...")
        
        # Stream through all previous segment files to get primes for pre-sieving
        current_prime = 2
        current_prime_index = 1
        primes_processed = 0
        
        for prev_file in previous_segment_files:
            with SegmentFile(data_directory, segment_size_billions, 0, mode='r') as sf:
                sf.filename = prev_file
                sf.file_handle = open(prev_file, 'rb')
                sf._read_header()  # Skip header
                
                # Start with prime 3 (we already handled 2 in initialization)
                if current_prime == 2:
                    current_prime = 3
                    current_prime_index = 2
                    primes_processed += 1
                
                for sieving_prime, sieving_prime_index in sf.iter_primes(current_prime, current_prime_index):
                    # Find where this prime first hits in the current segment
                    p_offset = sieving_prime - (segment_start % sieving_prime)
                    if p_offset == sieving_prime:
                        p_offset = 0
                    sieving_prime_index = min(sieving_prime_index, MAX_IDX) << 4
                    # Sieve multiples of this prime through the segment
                    while p_offset < segment_size:
                        max_factor_index_with_omega = segment_array[p_offset]
                        omega = min((max_factor_index_with_omega & 0x0F) + 1, 0x0F)
                        segment_array[p_offset] = sieving_prime_index | omega
                        p_offset += sieving_prime
                    
                    current_prime = sieving_prime
                    current_prime_index = sieving_prime_index
                    primes_processed += 1
                    
                    # Progress indicator for long pre-sieving operations
                    if primes_processed % 10_000_000 == 0:
                        print(f"    Pre-sieved with {primes_processed:,} primes, current prime: {sieving_prime:,}")
        
        print(f"  Pre-sieving complete. Used {primes_processed:,} primes.")
    
    # Discover new primes within this segment, sieve them, and test after each discovery
    search_start = 3 if segment_start == 0 else 0
    
    print(f"  Discovering new primes in segment...")
    primes_found_this_segment = 0
    
    while True:
        next_prime = 0
        
        # Find the next unmarked index -> prime in this block
        for i in range(search_start, segment_size):
            if segment_array[i] == 0:
                next_prime = segment_start + i
                break
        
        # If none left in this block, stop.
        if next_prime == 0:
            break

        gap_size = next_prime - last_prime_stored        
        next_to_last_prime = last_prime_stored
        last_prime_stored = next_prime
        prime_count += 1
        primes_found_this_segment += 1
        prime_idx = min(prime_count, MAX_IDX) << 4 # Once the index hits MAX_IDX, that's it.
        # No Pi-Complete number in any reachable goal can have a max index near this.

        if next_prime < prime_storing_limit:  # We stop storing primes when half way to goal
            delta = gap_size >> 1  # All deltas for odd primes are even, so save a bit
            segment_file.append_delta(delta)

        # Sieve multiples of this new prime within the current buffer
        # Only if 2*next_prime <= segment_start + segment_size
        if 2 * next_prime <= segment_start + segment_size:
            for n in range(next_prime, segment_start + segment_size, next_prime):
                n_offset = n - segment_start
                if n_offset >= segment_size:
                    break
                max_factor_index_with_omega = segment_array[n_offset]
                omega = min((max_factor_index_with_omega & 0x0F) + 1, 0x0F)
                segment_array[n_offset] = prime_idx | omega

        # Advance search to the next odd after this found prime
        search_start = last_prime_stored + 2 - segment_start
        
        # TEST the prime free space ending at this prime.
        prime_free_length = last_prime_stored - next_to_last_prime
            
        if prime_free_length > 2:  # Do testing if prime gap bigger than 2
            tuple_start = next_to_last_prime + 1    # Tuple must start at start of prime free space
            tuple_start_offset = tuple_start - segment_start

            for i in range(min(prime_free_length - 2, MAX_TUPLE - 1)):  # Test 2 to max integers
                # i = tuple size to try - 2 (zero for duplets, 1 for triplets, etc.)
                max_factor_index_with_omega = segment_array[tuple_start_offset]
                omega_sum = max_factor_index_with_omega & 0x0F  # Start summing with the first tuple omega value
                max_factor_idx = max_factor_index_with_omega >> 4

                for k in range(1, i + 2):
                    # k = number of multiplications to simulate by adding factor counts
                    max_factor_index_with_omega = segment_array[tuple_start_offset + k]
                    omega_sum += max_factor_index_with_omega & 0x0F  # Sum up the rest of the tuple omegas
                    max_factor_idx = max(max_factor_idx, (max_factor_index_with_omega >> 4))
                            
                if max_factor_idx == MAX_IDX:
                    continue  # Product can't have this high
                            
                # Adjust for shared prime factors
                if i > 0:
                    omega_sum -= correction_matrix[i][tuple_start %divisor_period]

                if omega_sum >= MAX_IDX:
                    continue  # Sum can't be this high

                # omega_sum now holds the omega value of the product of consecutive integers        
                if omega_sum == max_factor_idx:  # if that is the index of their greatest factor, win!
                    print(f"  Found pi-complete {i+2} integer product starting at {tuple_start:,}", end="")
                    found_list.append((tuple_start, i+2))
                                
                    # Check if it's a factorial (almost never happens, just calculate it)
                    product = tuple_start
                    for k in range(1, i + 2):
                        product *= tuple_start + k

                    # Verify result by SYMPY
                    if not is_pi_complete(product):
                        exit(f"ERROR -- {product=:,} not pi-complete for string starting at {tuple_start:,} and length {i+1}")

                    a, a_factorial = 1, 1
                    while a_factorial < product and a < 100:
                        a += 1
                        a_factorial *= a
                                
                    if a_factorial == product:
                        print(f" = {a}! Found factorial!")
                    elif has_FEF(product):
                        print(f" and has FEF.")
                    else:
                        print(".")

            if prime_free_length > MAX_TUPLE: # If tuple is too big to do common prime factor loookup
                # Check each integer in the tuple to see if we have already hit the max factor limit
                for i in range(MAX_TUPLE, prime_free_length): 
                    max_max, omega_sum = 0, 0
                    for k in range(i):
                        max_factor_index_with_omega = segment_array[tuple_start_offset + k]
                        max_max = max(max_max, (max_factor_index_with_omega >> 4))
                        omega_sum += max_factor_index_with_omega & 0x0F
                    if max_max < MAX_IDX:
                        print(f"MAX_IDX not hit in tupel length {i+1} at {tuple_start:,} with {prime_free_length=:,} and {omega_sum=:,}")
                
                
        # Progress indicator
        if primes_found_this_segment % 100_000_000 == 0:
            print(f"    Found {primes_found_this_segment:,} primes in this segment, current: {next_prime:,}")
        
        # Stop if we've reached the goal
        if next_prime >= goal:
            break

    print(f"  Segment {segment_number} complete: found {primes_found_this_segment:,} new primes")
    print(f"  Total primes so far: {prime_count:,}, last prime: {last_prime_stored:,}")
    
    return found_list, last_prime_stored, prime_count


def main():
    segment_size_billions = segment_size // 1_000_000_000    
    print("=" * 70)
    print("Pi-Complete Number Search with SSD-Paged Prime Storage")
    print("=" * 70)
    print(f"Goal: {goal:,}")
    print(f"Segment size: {segment_size:,} ({segment_size_billions} billion)")
    print(f"Data directory: {data_directory}")
    print()

    # Create data directory if it doesn't exist
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        print(f"Created data directory: {data_directory}")
    
    # Clean up any incomplete temp files from previous runs
    cleanup_temp_files(data_directory, segment_size_billions)

    # Scan for existing segment files
    existing_segments = scan_existing_segments(data_directory, segment_size)
    
    # ... rest of main() continues unchanged ...
    # Scan for existing segment files
    existing_segments = scan_existing_segments(data_directory, segment_size)
    
    # Determine starting point
    if existing_segments:
        last_seg_num, last_seg_file = existing_segments[-1]
        metadata = load_segment_metadata(last_seg_file, segment_size_billions)
        
        print(f"Found {len(existing_segments)} existing segment file(s)")
        print(f"Last completed segment: {last_seg_num}")
        print(f"  Last prime stored: {metadata['segment_end']:,}")
        print(f"  Total primes so far: {metadata['prime_count']:,}")
        
        # Resume from next segment
        start_segment = last_seg_num + 1
        last_prime_stored = metadata['segment_end']
        prime_count = metadata['prime_count']
        
        if last_prime_stored >= goal:
            print(f"\nGoal of {goal:,} already reached!")
            print("Exiting without additional computation.")
            return
        
        print(f"\nResuming from segment {start_segment}...")
    else:
        print("No existing segment files found. Starting fresh from segment 0...")
        start_segment = 0
        last_prime_stored = 2  # Pretend we stored the number 2 (the first prime)
        prime_count = 1  # Pretend we found just the first prime

    # Initialize working array
    segment_array = array.array('B', [0] * segment_size)
    
    # Collect all results
    all_results = []
    
    # Main segmented sieve loop
    segment_number = start_segment
    
    
    while last_prime_stored < goal:
        # Get list of previous segment files for pre-sieving
        previous_files = [filename for _, filename in existing_segments]

        # Create new segment file
        with SegmentFile(data_directory, segment_size_billions, segment_number, mode='w') as seg_file:
            # Build this segment
            segment_start = 0 if last_prime_stored < 3 else last_prime_stored + 1
            
            found, last_prime_stored, prime_count = build_segment(
                segment_number, segment_size, segment_array, goal,
                seg_file, previous_files,
                last_prime_stored, prime_count
            )
            
            all_results.extend(found)
            
            # Set metadata before closing file
            seg_file.set_metadata(segment_size, segment_start, last_prime_stored, prime_count)
        
        # Add this segment to the list of existing segments
        new_filename = os.path.join(
            data_directory,
            f"primes_{segment_size_billions}B_seg{segment_number:04d}.dat"
        )
        existing_segments.append((segment_number, new_filename))
        
        print(f"Segment {segment_number} file saved: {new_filename}")
        
        segment_number += 1
    
    # Final summary
    print("\n" + "=" * 70)
    print("SEARCH COMPLETE")
    print("=" * 70)
    print(f"Goal reached: {goal:,}")
    print(f"Total segments computed: {segment_number}")
    print(f"Total primes found: {prime_count:,}")
    print(f"Last prime: {last_prime_stored:,}")
    print(f"\nFound {len(all_results):,} pi-complete products of composite consecutive integers.")
    
    if all_results:
        print("\nPi-complete products found:")
        for start, length in all_results:
            print(f"  {length} integers starting at {start:,}")


if __name__ == "__main__":
    main()
    print("\nEnd of Program")
