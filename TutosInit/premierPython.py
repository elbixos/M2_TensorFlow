
def addition (a, b):
    resu = a+b;
    return resu


print ('Bonjour a tous');

x = 5
print(x)
print(type(x))


tab = [5,7,"toto",3.3]
print (tab)

if x < 7:
    print ("petit")
    print (" nombre")
else :
    print ("grand")
    print (" nombre")

print ("au revoir")

tab[1] = 0

tab.append(24)
tab.append("truc")

print ("Parcours de tableau a la mode python")
for elt in tab:
    print (elt)
    
print ("Parcours de tableau indexe par l'indice")  
for i in range(len(tab)):
    print (tab[i])     

print("Appel de fonction")
y = 7
calcul = addition (x,y)
print(calcul)