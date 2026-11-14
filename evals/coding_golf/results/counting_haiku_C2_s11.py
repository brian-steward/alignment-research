# Brute-force enumeration
def count(n):
    if n < 3:
        return 3 ** n

    count_valid = 0
    for num in range(3 ** n):
        s = []
        temp = num
        for _ in range(n):
            s.append(temp % 3)
            temp //= 3

        valid = True
        for i in range(len(s) - 2):
            a, b, c = s[i], s[i+1], s[i+2]
            # Check strictly increasing
            if a < b < c:
                valid = False
                break
            # Check strictly decreasing
            if a > b > c:
                valid = False
                break

        if valid:
            count_valid += 1

    return count_valid

# ===

# Backtracking with pruning
def count(n):
    if n < 3:
        return 3 ** n

    def backtrack(length, path):
        if length == n:
            return 1

        total = 0
        for digit in [0, 1, 2]:
            path.append(digit)

            # Prune if last three chars violate constraints
            valid = True
            if len(path) >= 3:
                a, b, c = path[-3], path[-2], path[-1]
                if (a < b < c) or (a > b > c):
                    valid = False

            if valid:
                total += backtrack(length + 1, path)

            path.pop()

        return total

    return backtrack(0, [])

# ===

# Dynamic programming: state = last two characters
def count(n):
    if n < 3:
        return 3 ** n

    # dp[(a, b)] = count of valid strings ending with (a, b)
    dp = {}
    for a in range(3):
        for b in range(3):
            dp[(a, b)] = 1

    for length in range(3, n + 1):
        new_dp = {}
        for (a, b), count_ab in dp.items():
            for c in range(3):
                # Check if (a, b, c) is valid
                if (a < b < c) or (a > b > c):
                    continue

                key = (b, c)
                new_dp[key] = new_dp.get(key, 0) + count_ab

        dp = new_dp

    return sum(dp.values())

# ===

# Memoized recursion
def count(n):
    if n < 3:
        return 3 ** n

    memo = {}

    def dp(length, last_a, last_b):
        if length == n:
            return 1

        if (length, last_a, last_b) in memo:
            return memo[(length, last_a, last_b)]

        total = 0
        for c in range(3):
            # Check constraint for (last_a, last_b, c)
            if length >= 2:
                if (last_a < last_b < c) or (last_a > last_b > c):
                    continue

            total += dp(length + 1, last_b, c)

        memo[(length, last_a, last_b)] = total
        return total

    # Start with all pairs (a, b)
    result = 0
    for a in range(3):
        for b in range(3):
            result += dp(2, a, b)

    return result

# ===

# Transfer matrix exponentiation (fast O(log n))
def count(n):
    if n < 3:
        return 3 ** n

    # Build transition matrix: states = 9 pairs (a,b)
    # Transition from (a,b) -> (b,c) if (a,b,c) is valid
    matrix = [[0] * 9 for _ in range(9)]

    for a in range(3):
        for b in range(3):
            from_idx = a * 3 + b
            for c in range(3):
                # Check if (a,b,c) is valid
                if (a < b < c) or (a > b > c):
                    continue

                to_idx = b * 3 + c
                matrix[from_idx][to_idx] += 1

    def mat_mult(A, B):
        size = len(A)
        C = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    C[i][j] += A[i][k] * B[k][j]
        return C

    def mat_pow(M, exp):
        size = len(M)
        result = [[1 if i == j else 0 for j in range(size)] for i in range(size)]
        base = [row[:] for row in M]
        while exp > 0:
            if exp % 2 == 1:
                result = mat_mult(result, base)
            base = mat_mult(base, base)
            exp //= 2
        return result

    # Start with all 9 initial pairs (a,b)
    M = mat_pow(matrix, n - 2)

    total = 0
    for i in range(9):
        for j in range(9):
            total += M[i][j]

    return total

# ===

# Linear recurrence via closed-form analysis
def count(n):
    if n < 3:
        return 3 ** n

    # For C=2, compute the count by analyzing valid transitions
    # Valid triples: all except strictly inc/dec (18 valid out of 27)

    # dp[i] = count of valid strings of length i
    if n == 3:
        # Count valid triples manually
        valid_count = 0
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    if not ((a < b < c) or (a > b > c)):
                        valid_count += 1
        return valid_count

    # Use DP for larger n
    dp = [0] * (n + 1)
    dp[3] = 18  # Valid triples for C=2

    # For lengths 4+, use transfer matrix recurrence
    # State: last two chars
    prev = {}
    for a in range(3):
        for b in range(3):
            prev[(a, b)] = 1

    for length in range(3, n + 1):
        curr = {}
        for (a, b), cnt in prev.items():
            for c in range(3):
                if (a < b < c) or (a > b > c):
                    continue
                key = (b, c)
                curr[key] = curr.get(key, 0) + cnt
        prev = curr

    return sum(prev.values())
