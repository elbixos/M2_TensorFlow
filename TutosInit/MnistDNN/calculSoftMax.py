import math


score = [2,2,3,2]

score2 = []
somme = 0
for s in score :
    temp = math.exp(s)
    score2.append(temp)
    print (temp)
    somme+=temp

print (somme)
print (score2)

soft = []
for s in score2 :
    soft.append(s / somme)

print (soft)