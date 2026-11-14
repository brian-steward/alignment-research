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
            # Constraint 1: not strictly increasing
            if a < b < c:
                valid = False
                break
            # Constraint 2: not strictly decreasing
            if a > b > c:
                valid = False
                break
            # Constraint 3: not all equal
            if a == b == c:
                valid = False
                break
            # Constraint 4: (a+b+c) mod 3 != 2
            if (a + b + c) % 3 == 2:
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

    def is_valid_triple(a, b, c):
        if (a < b < c) or (a > b > c):
            return False
        if a == b == c:
            return False
        if (a + b + c) % 3 == 2:
            return False
        return True

    def backtrack(path):
        if len(path) == n:
            return 1

        total = 0
        for digit in [0, 1, 2]:
            path.append(digit)

            valid = True
            if len(path) >= 3:
                a, b, c = path[-3], path[-2], path[-1]
                if not is_valid_triple(a, b, c):
                    valid = False

            if valid:
                total += backtrack(path)

            path.pop()

        return total

    return backtrack([])

# ===

# Dynamic programming: state = last two characters
def count(n):
    if n < 3:
        return 3 ** n

    def is_valid_triple(a, b, c):
        if (a < b < c) or (a > b > c):
            return False
        if a == b == c:
            return False
        if (a + b + c) % 3 == 2:
            return False
        return True

    # dp[(a, b)] = count of valid strings ending with (a, b)
    dp = {}
    for a in range(3):
        for b in range(3):
            dp[(a, b)] = 1

    for length in range(3, n + 1):
        new_dp = {}
        for (a, b), cnt in dp.items():
            for c in range(3):
                if not is_valid_triple(a, b, c):
                    continue

                key = (b, c)
                new_dp[key] = new_dp.get(key, 0) + cnt

        dp = new_dp

    return sum(dp.values())

# ===

# Memoized recursion
def count(n):
    if n < 3:
        return 3 ** n

    def is_valid_triple(a, b, c):
        if (a < b < c) or (a > b > c):
            return False
        if a == b == c:
            return False
        if (a + b + c) % 3 == 2:
            return False
        return True

    memo = {}

    def dp(length, last_a, last_b):
        if length == n:
            return 1

        if (length, last_a, last_b) in memo:
            return memo[(length, last_a, last_b)]

        total = 0
        for c in range(3):
            if length >= 2 and not is_valid_triple(last_a, last_b, c):
                continue

            total += dp(length + 1, last_b, c)

        memo[(length, last_a, last_b)] = total
        return total

    result = 0
    for a in range(3):
        for b in range(3):
            result += dp(2, a, b)

    return result

# ===

# Transfer matrix exponentiation
def count(n):
    if n < 3:
        return 3 ** n

    def is_valid_triple(a, b, c):
        if (a < b < c) or (a > b > c):
            return False
        if a == b == c:
            return False
        if (a + b + c) % 3 == 2:
            return False
        return True

    # Build transition matrix: 9x9 for states (a,b)
    matrix = [[0] * 9 for _ in range(9)]

    for a in range(3):
        for b in range(3):
            from_idx = a * 3 + b
            for c in range(3):
                if not is_valid_triple(a, b, c):
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

    M = mat_pow(matrix, n - 2)

    total = 0
    for i in range(9):
        for j in range(9):
            total += M[i][j]

    return total

# ===

# Iterative DP building transition table explicitly
def count(n):
    if n < 3:
        return 3 ** n

    def is_valid_triple(a, b, c):
        if (a < b < c) or (a > b > c):
            return False
        if a == b == c:
            return False
        if (a + b + c) % 3 == 2:
            return False
        return True

    # Precompute valid transitions
    valid_transitions = {}
    for a in range(3):
        for b in range(3):
            valid_transitions[(a, b)] = []
            for c in range(3):
                if is_valid_triple(a, b, c):
                    valid_transitions[(a, b)].append(c)

    # DP
    dp = {}
    for a in range(3):
        for b in range(3):
            dp[(a, b)] = 1

    for _ in range(n - 2):
        new_dp = {}
        for (a, b), cnt in dp.items():
            for c in valid_transitions[(a, b)]:
                key = (b, c)
                new_dp[key] = new_dp.get(key, 0) + cnt
        dp = new_dp

    return sum(dp.values())
