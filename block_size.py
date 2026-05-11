##############################################################################################
# This is a file to define the function block_size()
# This will generate block sizes for every session and return the block size arrangement.
# Params to change: target_sum = 600; 2*N (line 31&34); 2*b_min, b_max (line 32&35)
#############################################################################################


from random import randint,shuffle


def block_size(group):
    '''
    A. Generate block sizes for every session. This depends on: 
    1) Which group is this participant in; 
    2) randomization of block sizes (40-60 in group 0 & 130-210 in group 1)
    group: 0 - short blocks group; 1 - long blocks group
    Methods:
    Generates N different block sizes (b1, b2... bN)
    Sum of bi = 600
    Group 0: N = 11, bi= [40,60]
    Group 1: N = 4, bi=[130,210];
    Change the rule;

    B. return the block size arrangement, ready to be writen into the main csv doc. 
    '''
    target_sum = 30 # total trail number
    max_attempts = 1000 # attempts to find series

    if group == 0: # short-block group
        N = 10 # N: number of blocks
        b_min, b_max = 1, 4
    else:
        N = 3
        b_min, b_max = 6, 12
    
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

    res[0] += max(0, d)
    res[1] -= max(0, d)
    
    shuffle(res)
    return res, N

## DEBUG
## group = int(input("enter group:"))
## print(block_size(group))
