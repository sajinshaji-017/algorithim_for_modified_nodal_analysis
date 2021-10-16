import os
from sympy import *
import numpy as np
import pandas as pd
init_printing()

num_rlc = 0
num_v = 0
num_i = 0
i_unk = 0
num_d = 0

fn = 'circuit_04'  # nestlist file name is added
fd1 = open(fn + '.net', 'r')
content = fd1.readlines()
content = [x.strip() for x in content]  # remove leading and trailing white space
# remove empty lines
while '' in content:
    content.pop(content.index(''))
# remove comment lines, these start with a asterisk *
content = [n for n in content if not n.startswith('*')]
# remove other comment lines, these start with a semicolon ;
content = [n for n in content if not n.startswith(';')]
# remove spice directives, these start with a period, .
content = [n for n in content if not n.startswith('.')]
# content = [x.upper() for x in content] <- this converts all to upper case
content = [x.capitalize() for x in content]
# removes extra spaces between entries
content = [' '.join(x.split()) for x in content]
#print(content)

# for counting elements in the circuit
line_cnt = len(content)  # number of lines in the netlist
#print(line_cnt)
branch_cnt = 0  # number of branches in the netlist
# check number of entries on each line, count each element type
for i in range(line_cnt):
    x = content[i][0]
    tk_cnt = len(content[i].split())  # split the line into a list of words
    # print(content[i].split())

    if (x == 'R') or (x == 'L') or (x == 'C'):
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}"
                  .format(i, content[i]))
            print("had {:d} items and should only be 4"
                  .format(tk_cnt))
        num_rlc += 1
        branch_cnt += 1
    elif x == 'V':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}"
                  .format(i, content[i]))
            print("had {:d} items and should only be 4"
                  .format(tk_cnt))
        num_v += 1
        branch_cnt += 1
    elif x == 'I':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}"
                  .format(i, content[i]))
            print("had {:d} items and should only be 4"
                  .format(tk_cnt))
        num_i += 1
        branch_cnt += 1
    elif x == 'D':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}"
                  .format(i, content[i]))
            print("had {:d} items and should only be 4"
                  .format(tk_cnt))
        num_d += 1
        branch_cnt += 1


    else:
        print("unknown element type in branch {:d}, {:s}"
              .format(i, content[i]))



# build the pandas data frame
df = pd.DataFrame(columns=['element', 'p node', 'n node', 'value'])
# this data frame is for branches with unknown currents
df2 = pd.DataFrame(columns=['element', 'p node', 'n node'])


def indep_source(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu, 'element'] = tk[0]
    df.loc[line_nu, 'p node'] = int(tk[1])
    df.loc[line_nu, 'n node'] = int(tk[2])
    df.loc[line_nu, 'value'] = str(tk[3])


# loads passive elements into branch structure
def rlc_element(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu, 'element'] = tk[0]
    df.loc[line_nu, 'p node'] = int(tk[1])
    df.loc[line_nu, 'n node'] = int(tk[2])
    df.loc[line_nu, 'value'] = str(tk[3])


# loads active element into branch structure
def diode_element(line_nu):
    tk = content[line_nu].split()
    df.loc[line_nu, 'element'] = tk[0]
    df.loc[line_nu, 'p node'] = int(tk[1])
    df.loc[line_nu, 'n node'] = int(tk[2])
    df.loc[line_nu, 'value'] = str(tk[3])



# function to scan df and get largest node number
def count_nodes():
    # need to check that nodes are consecutive
    # fill array with node numbers
    p = np.zeros(line_cnt+1)
    for j in range(line_cnt-1):
        p[df['p node'][j]] = df['p node'][j]
        p[df['n node'][j]] = df['n node'][j]

        # find the largest node number
        if df['n node'].max() > df['p node'].max():
            largest = df['n node'].max()
        else:
            largest = df['p node'].max()

        largest = int(largest)

        return largest


# load branch info into data frame
for i in range(line_cnt):
    x = content[i][0]

    if (x == 'R') or (x == 'L') or (x == 'C'):
        rlc_element(i)
    elif (x == 'V') or (x == 'I'):
        indep_source(i)
    elif x == 'D':
        diode_element(i)
    else:
        print("unknown element type in branch {:d}, {:s}"
              .format(i, content[i]))


# count number of nodes
num_nodes = count_nodes()

# Build df2: consists of branches with current unknowns, used for C & D matrices
# walk through data frame and find these parameters

count = 0
for i in range(len(df)):
    # process all the elements creating unknown currents
    x = df.loc[i, 'element'][0]   # get 1st letter of element name
    if x == 'V':
        df2.loc[count, 'element'] = df.loc[i, 'element']
        df2.loc[count, 'p node'] = df.loc[i, 'p node']
        df2.loc[count, 'n node'] = df.loc[i, 'n node']
        count += 1

# print a report
print('Net list report')
print('number of lines in netlist: {:d}'.format(line_cnt))
print('number of branches: {:d}'.format(branch_cnt))
print('number of nodes: {:d}'.format(num_nodes))

# count the number of element types that affect the size of the B, C, D, E and J arrays
# these are current unknows

i_unk = num_v
print('number of unknown currents: {:d}'.format(i_unk))
print('number of RLC (passive components): {:d}'.format(num_rlc))
print('number of independent voltage sources: {:d}'.format(num_v))
print('number of independent current sources: {:d}'.format(num_i))
print('number of diodes: {:d}'.format(num_d))
print()
print(df)
print(df2)
print()

# initialize some symbolic matrix with zeros
# A is formed by [[G, C] [B, D]]
# Z = [I,Ev]
# X = [V, J]
V = np.zeros((num_nodes, 1), dtype=object)
I = np.zeros((num_nodes, 1), dtype=object)
G = np.zeros((num_nodes, num_nodes), dtype=object)
s = Symbol('s')  # the Laplace variable

# count the number of element types that affect the size of the B, C, D, E and J arrays
# these are current unknows
i_unk = num_v
if i_unk != 0:  # just generate empty arrays
    B = np.zeros((num_nodes, i_unk), dtype=object)
    C = np.zeros((i_unk, num_nodes), dtype=object)
    D = np.zeros((i_unk, i_unk), dtype=object)
    Ev = np.zeros((i_unk, 1), dtype=object)
    J = np.zeros((i_unk, 1), dtype=object)

# G matrix
for i in range(len(df)):  # process each row in the data frame
    n1 = df.loc[i, 'p node']
    n2 = df.loc[i, 'n node']
    # process all the passive elements, save conductance to temp value
    x = df.loc[i, 'element'][0]   # get 1st letter of element name
    if x == 'R':
        g = 1/sympify(df.loc[i, 'element'])
    if x == 'C':
        g = s*sympify(df.loc[i, 'element'])
    if x == 'L':
        g = 1/s/sympify(df.loc[i, 'element'])
    if x == 'D':
        g = sympify('G{:s}'.format(df.loc[i, 'element'][1]))



    if (x == 'R') or (x == 'C') or (x == 'L') or (x == 'D'):
        # If neither side of the element is connected to ground
        # then subtract it from appropriate location in matrix.
        if (n1 != 0) and (n2 != 0):
            G[n1-1, n2-1] += -g
            G[n2-1, n1-1] += -g

        # If node 1 is connected to ground, add element to diagonal of matrix
        if n1 != 0:
            G[n1-1, n1-1] += g

        # same for for node 2
        if n2 != 0:
            G[n2-1, n2-1] += g

print("G Matrix:", np.array2string(G, prefix="G Matrix:",
                                   separator=", "))
print()


#   B matrix
sn = 0
for i in range(len(df2)):
    n1 = df2.loc[i, 'p node']
    n2 = df2.loc[i, 'n node']

    x = df2.loc[i, 'element'][0]
    if x == 'V':
        if i_unk > 1:
            if n1 != 0:
                B[n1 - 1, sn] = 1
            if n2 != 0:
                B[n2 - 1, sn] = -1
        else:
            if n1 != 0:
                B[n1 - 1] = 1
            if n2 != 0:
                B[n2 - 1] = -1
        sn += 1

if sn != i_unk:
    print("source number, sn={:d} not equal to i_unk={:d} in matrix B "
          .format(sn, i_unk))

print("B Matrix:", np.array2string(B, prefix="B Matrix:",
                                   separator=", "))
print()

# C matrix
sn = 0
for i in range(len(df2)):
    n1 = df2.loc[i, "p node"]
    n2 = df2.loc[i, 'n node']

    x = df2.loc[i, 'element'][0]
    if x == 'V':
        if i_unk > 1:
            if n1 != 0:
                C[sn, n1 - 1] = 1
            if n2 != 0:
                C[sn, n2 - 1] = -1

        else:
            if n1 != 0:
                C[0, n1 - 1] = 1
            if n2 != 0:
                C[0, n2 - 1] = -1

        sn += 1

if sn != i_unk:
    print("Source Count, sn={:d} not equal to i_unk={:d} in matrix C "
          .format(sn, i_unk))

print("C Matrix:", np.array2string(C, prefix="C Matrix:",
                                   separator=", "))
print()

print("D matrix:", np.array2string(D, prefix="D Matrix:",
                                   separator=", "))
print()

# Augmented Matrix
n = num_nodes
m = i_unk
A = np.zeros((n+m, n+m), dtype=object)
for i in range(n):
    for j in range(n):
        A[i, j] = G[i, j]

if i_unk > 1:
    for i in range(n):
        for j in range(m):
            A[i, n + j] = B[i, j]
            A[n + j, i] = C[j, i]

    for i in range(m):
        for j in range(m):
            A[n + i, n + j] = D[i, j]

else:
    for i in range(n):
        A[i, n] = B[i, 0]
        A[n, i] = C[0, i]

print("A Matrix:", np.array2string(np.asmatrix(A), prefix="A Matrix:",
                                   separator=", "))
print()

# generation of unknown vector matrix[V, J]
# generation of V sub matrix
for i in range(num_nodes):
    V[i] = sympify('v{:d}'.format(i + 1))

print("V matrix:", np.array2string(V, prefix="V matrix:",
                                   separator=", "))
print()

# generate J sub matrix
for i in range(len(df2)):
    J[i] = sympify('I{:s}'.format(df2.loc[i, 'element']))

print("J matrix:", np.array2string(J, prefix="J matrix:",
                                   separator=", "))
print()

# X matrix
n = num_nodes
m = i_unk
X = np.zeros((n + m, 1), dtype=object)
for i in range(n):
    X[i] = V[i]

for i in range(m):
    X[n + i] = J[i]

print("X matrix:", np.array2string(X, prefix="X matrix:",
                                   separator=", "))

print()

# Generation of Z matrix[I, Ev]
# generation of I sub matrix
for i in range(len(df)):
    n1 = df.loc[i, 'p node']
    n2 = df.loc[i, 'n node']
    x = df.loc[i, 'element'][0]

    if x == "I":
        g = sympify(df.loc[i, 'element'])
        if n1 != 0:
            I[n1 - 1] -= g
        if n2 != 0:
            I[n2 - 1] += g

    elif x == 'D':
        g = sympify('Id{:s}'.format(df.loc[i, 'element'][1]))
        if n1 != 0:
            I[n1 - 1] -= g
        if n2 != 0:
            I[n2 - 1] += g

print("I matrix:", np.array2string(I, prefix="I matrix:",
                                       separator=","))
print()

# Generate sub matrix Ev
sn = 0
for i in range(len(df2)):
    x = df2.loc[i, 'element'][0]
    if x == "V":
        Ev[sn] = sympify(df2.loc[i, 'element'])
        sn += 1

print("Ev matrix:", np.array2string(Ev, prefix="Ev matrix:",
                                    separator=","))
print()

n = num_nodes
m = i_unk
Z = np.zeros((n + m, 1), dtype=object)
for i in range(n):
    Z[i] = I[i]

for i in range(m):
    Z[n + i] = Ev[i]

print("Z matrix:", np.array2string(Z, prefix="Z matrix:",
                                   separator=", "))