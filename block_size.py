##############################################################################################
# define function block_size()
# This will generate block sizes for every session and return the block size arrangement.
#############################################################################################


from random import randint, shuffle


def block_size(group):
    '''
    A. Generate block sizes for every session. This depends on: 
    1) Which group is this participant in; 
    2) randomization of block sizes (40-60 in group 0 & 150-250 in group 1)
    group: 0 - short blocks group; 1 - long blocks group
    Methods:
    Generates N different block sizes (b1, b2... bN)
    Sum of bi = 1200
    Group 0: N = 24, bi= [40,60]
    Group 1: N = 6, bi=[150,250];

    B. return the block size arrangement, ready to be written into the main csv doc. 
    '''
    target_sum = 1200 # total trial number
    max_attempts = 1000 # attempts to find series

    if group == 0: # short-block group
        N = 24 # N: number of blocks
        b_min, b_max = 40, 60
    else:
        N = 6
        b_min, b_max = 150, 250
    
    # Try to generate series that satisfy constraints
    for attempt in range(max_attempts):
        blocks = []
        current_sum = 0
        for i in range(N - 1):
            remain_n = N - 1 - i
            low = max(b_min, target_sum - current_sum - remain_n * b_max)
            high = min(b_max, target_sum - current_sum - remain_n * b_min)

            if low > high:
                break
            
            b_i = randint(int(low), int(high))
            blocks.append(b_i)
            current_sum += b_i
        # Calculate bN when loop not break
        if len(blocks) == N - 1:
            last_b = target_sum - current_sum
            # check constraint for bN
            if b_min <= last_b <= b_max:
                blocks.append(last_b)
            
                if attempt < max_attempts*2//3: # no equal block sizes
                    if len(set(blocks)) == len(blocks):
                        shuffle(blocks)
                        return blocks, N
                else: # accept some equal block sizes
                    shuffle(blocks)
                    return blocks, N

    # very unlikely case
    avg, rem = divmod(target_sum, N)
    res = [avg + 1 if i < rem else avg for i in range(N)]

    d = int(min(avg - b_min, b_max - avg) - 2)

    delta = max(0, d)
    actual_delta = min(delta, res[1] - b_min)

    res[0] += actual_delta
    res[1] -= actual_delta
    
    shuffle(res)
    return res, N

## DEBUG
# group = int(input("enter group:"))
# print(block_size(group))
