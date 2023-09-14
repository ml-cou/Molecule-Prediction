import numpy as np
import pandas as pd

# User-set parameters
identity_position=[0,1]
bead_position=[1,1]
bond_position=[2,1]
output_directory_node_features='Output/Node_Features'
output_directory_adjacency='Output/Adjacency_Matrix'

# Import data
lipid_information=pd.read_table('Python-Input.txt',delimiter='\t',header=None)
atom_identity_input=pd.read_table('bead_info-3.txt',delimiter='\t',header=None)

input_array=lipid_information.to_numpy()
bead_identities=atom_identity_input.to_numpy()


#Create nested list for each Lipid
list_index=0
lipids=[]
addme=[]
for element in input_array:
    element=element.tolist()
    addme.append(element)
    if element[0] == 'BONDS':
        list_index+=1
        lipids.append(addme)
        addme=[]


#print(lipids)
#print(atom_identity_input)
#print(bead_identities[0])
#print(bead_identities[0][2])


# Find latest indecies of tails starting
def latest_tail_index(lipids):
    latest_A_start=0
    latest_B_start=0

    # Split node identites into separate elements
    for lipid in lipids:

        nodes=str(lipid[1][1]).split(' ')

        # If ending element is empty, remove it.
        if nodes[-1]=='':
            nodes=nodes[:-1]

        # Enumerate through nodes, if a tail starts later than the latest current recorded one, replace that one with the new enumeration number
        for enum,node in enumerate(nodes):
            if '1A' in node and enum>latest_A_start:
                latest_A_start=enum

            if '1B' in node and enum>latest_B_start:
                latest_B_start=enum

    return(latest_A_start,latest_B_start)


# Find longest tail lengths
def longest_tail_length(lipids):
    longest_A_length=0
    longest_B_length=0
    count_longest = 0
    for lipid in lipids:
        count_A=0
        count_B=0

        nodes = str(lipid[1][1]).split(' ')

        # If ending element is empty, remove it.
        if nodes[-1] == '':
            nodes = nodes[:-1]

        # Iterate through nodes, increasing the count of nodes in A and B tails when found.
        for enum,node in enumerate(nodes):
            if node[-1]=='A':
                count_A+=1

            if node[-1]=='B':
                count_B+=1

            if enum>count_longest:
                count_longest=enum

        # If count of current lipids tail length is longer than existing, replace it.
        if count_A>longest_A_length:
            longest_A_length=count_A

        if count_B>longest_B_length:
            longest_B_length=count_B

    return (longest_A_length,longest_B_length,count_longest)


# Find the unique beads present in atom_identity_input
def find_unique_beads(bead_information):
    bead_types=[]

    for line in bead_information:

        if line[1] not in bead_types:
            bead_types.append(line[1])

    # print(bead_types)
    return(bead_types)


# Generate 1-hot node feature matrix and write it.
def node_features(lipid,bead_types,bead_identities,start_A,start_B,longest_lipid):
    # Function input data processing
    #print(lipid)
    identity=lipid[identity_position[0]][identity_position[1]]
    if identity[-1]==' ':
        identity=identity[:-1]

    nodes=lipid[bead_position[0]][bead_position[1]]
    nodes=nodes.split(' ')
    if nodes[-1]=='':
        nodes=nodes[:-1]

    #print(nodes)

    node_matrix=np.zeros((longest_lipid,len(bead_types))) #Zero array length of longest lipid , by number of bead types.

    # Generation of row(Bead identity) and column headers (Bead types)
    column_headers=[]
    for bead_type in bead_types:
        column_headers.append(bead_type)

    row_headers=np.zeros(longest_lipid)
    row_headers=list(row_headers)


    current_index=0
    #print(identity)
    # Find the start index of the lipid in bead_identites
    for enum,bead in enumerate(bead_identities):
        if bead[3]==identity:
            lipid_bead_identity_start=enum
            break

    #Iterate through each node, and manipulate the node feature matrix for a given nodes bead type.
    for node in nodes:
        #print(node)
        if '1A' in node:
            current_index=start_A

        if '1B' in node:
            current_index=start_B

        for bead in bead_identities[lipid_bead_identity_start:]:
            if bead[4]==node and bead[3]==identity:
                bead_type=bead[1]
                #print(bead_type)

        row_headers[current_index]=node
        node_matrix[current_index][bead_types.index(bead_type)]=1

        current_index+=1

    # Replace all 0 elements in row_headers with -
    while 0.0 in row_headers:
        row_headers[row_headers.index(0.0)]='-'

    #print(node_matrix)

    # Create pandas dataframe
    node_DF=pd.DataFrame(node_matrix)
    node_DF.columns=column_headers
    node_DF.index=row_headers

    #print(node_DF)

    write_me('Node',identity,node_DF)

    return(row_headers)


def adjacency_matrix(lipid,nodes):
    # Bond data pre-processing
    bonds=lipid[bond_position[0]][bond_position[1]]
    identity = lipid[identity_position[0]][identity_position[1]]
    bonds=bonds.split(' ')
    if bonds[-1]=='':
        bonds=bonds[:-1]

    adjacency_array=np.zeros((len(nodes),len(nodes)))  #Adjacency matrix generation
    # Update adjacency matrix using each bond.
    for bond in bonds:
        bonded_beads=bond.split('-')
        bead_1_index=nodes.index(bonded_beads[0])
        bead_2_index=nodes.index(bonded_beads[1])

        adjacency_array[bead_1_index][bead_2_index] = 1
        adjacency_array[bead_2_index][bead_1_index] = 1

    # Dataframe generation and writing
    print(len(nodes),len(adjacency_array[0]))
    adjacency_dataframe=pd.DataFrame(adjacency_array)
    adjacency_dataframe.columns=nodes
    adjacency_dataframe.index=nodes

    #print(adjacency_dataframe)
    write_me('Adjacency',identity,adjacency_dataframe)


def write_me(type,identity,data):
    if type=='Adjacency': data.to_csv(f'{output_directory_adjacency}/{identity}.txt',sep='\t')
    if type=='Node': data.to_csv(f'{output_directory_node_features}/{identity}.txt',sep='\t')


#Find the latest start positions for each tail, and the longest tails.
latest_A,latest_B=latest_tail_index(lipids)
longest_A,longest_B,longest_lipid=longest_tail_length(lipids)
bead_types=find_unique_beads(bead_identities) #Find the unique bead types in the bead identities.
# print(latest_A,latest_B)
#print(longest_A,longest_B,longest_lipid)

# Standardized start positions for tails and head groups.
start_A=latest_A

# Manipulate start positions and longest lipid length based on tail lengths.
if start_A+longest_A>latest_B:
    start_B=start_A+longest_A
else:
    start_B=latest_B

if start_B+longest_B>longest_lipid:
    longest_lipid=start_B+longest_B

# Generate and write the node and adjacency matricies for each lipid.
for lipid in lipids:
    nodes=node_features(lipid,bead_types,bead_identities,start_A,start_B,longest_lipid)
    adjacency_matrix(lipid,nodes)