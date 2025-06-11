n = int(input())  # the number of temperatures to analyse
"""
the first comes -1 or 1  will be save as closest as 0
4
-2 1 3 5 
"""
closest = None

if n != 0: 
    for i in input().split():

        current = int(i)
        if current is None:
            print(0)
        else:
            if (closest is None or abs(closest) > abs(current)
                    or (abs(closest) == abs(current)
                        and closest < current)):
                    closest = current


    print(closest)
else: 
    print(0)
