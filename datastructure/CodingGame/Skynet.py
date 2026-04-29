import math

# -------------------------------------------------------------------------------------------------
# Skynet class: with a dictionary of all the nodes and all the gateways in the skynet
# -------------------------------------------------------------------------------------------------
class Skynet:

    def __init__(self, numnodes, numlinks, numgateways):

        self.size = numnodes
        self.numlinks = numlinks
        self.numgateways = numgateways
        self.gateways = {}
        self.nodes = {_: Node(_) for _ in range(0, self.size)}

    def addLink(self, pair):

        # We link the nodes and assign a weight to the link.
        # For this Skynet scenario weight will always be 1 but it mught be usefull in other scenarios
        self.nodes[pair[0]].links[pair[1]] = [pair[1], 1]
        self.nodes[pair[1]].links[pair[0]] = [pair[0], 1]
        # print([pair[0], 1])
        return

    def addGateway(self, gate):

        self.gateways[gate] = gate
        return

    # -------------------------------------------------------------------------------------------------
    # Method to find (dijkstra) all the shortest paths from a given node (gate) to all the other nodes
    # in the Skynet graph
    # -------------------------------------------------------------------------------------------------
    def shortestPaths(self, gate):

        # We create a dictionay of nodes (tablenodes) with a double value:
        #  1.- The size of the path (initialized to infinit)
        #  2.- The optimal path ("" to start with)
        # We also create a dictionary of visited nodes (visitednodes) where we keep the nodes not visited yet
        tablenodes = {}
        visitednodes = {}

        # We initialize both dictionaries
        for node in self.nodes.keys():
            tablenodes[node] = [math.inf, ""]
            visitednodes[node] = False

        # We apply initial values to the gateway node (0 distance and no path to itself)
        tablenodes[gate] = [0, str(gate) + ";"]

        # we delete all the gate nodes from the visited dict (except the one we are working with)
        # so we do not go through them when finding the different paths
        for restgates in self.gateways:
            if restgates != gate:
                del visitednodes[restgates]

        # We repeat the process while there are nodes in the visited dictionary
        while len(visitednodes) > 0:

            # We set the pointer to the next node with shorter path and that is still on the visited dict
            # For that, we go through the visitednodes dict. We call the node to work with: "minnode"
            # The first time it will select the gate node since it is the only one with value different from Inf
            minnode = ""
            for node in visitednodes.keys():
                if minnode == "":
                    minnode = node
                else:
                    if tablenodes[node][0] < tablenodes[minnode][0]:
                        minnode = node

            # We delete the selected node from the visited dict
            del visitednodes[minnode]

            # We analyse all the links associated to the minimum distance node
            for link in self.nodes[minnode].links.keys():

                # But we process only those nodes that have not been visited yet
                if link in visitednodes.keys():

                    # We calculate the distance: current distance to the node + the weight of the link (always 1)

                    # Note: For the size/distance of the path:
                    #    - we will add 1 if the node is not linked with any gateway
                    #    - we will add 0 the node if it has a link to a gate
                    #    - we will add -1 if the node is linked to two gates

                    # if the node shares a link with any gateway we substract one from "foundgateways"
                    foundgateways = 1
                    for thelinks in self.nodes[link].links:
                        if thelinks in self.gateways:
                            foundgateways -= 1

                    distance = tablenodes[minnode][0] + foundgateways

                    # If the difference is lower than the one registered on the nodes path table
                    # or if the distance is equal but the path is shorter.
                    # We update the nodes path table ("tablenodes") for the current node (link)
                    pathsizemin = len(tablenodes[minnode][1].split(";")[:-1])
                    pathsizelink = len(tablenodes[link][1].split(";")[:-1])

                    if distance < tablenodes[link][0] or \
                            (distance == tablenodes[link][0] and pathsizelink > pathsizemin):

                        tablenodes[link][0] = distance
                        tablenodes[link][1] = tablenodes[minnode][1] + str(link) + ";"

        # We exit the while loop if all the nodes have been visited.
        # The table of nodes path is now updated with all the
        # possible minimum paths. We return the node table.
        return tablenodes


    # -------------------------------------------------------------------------------------------------
    # Method to call the generic shortestPaths method for every single gateway
    # returning only the shortest path to the Agent containing the link thas has to be severed
    # -------------------------------------------------------------------------------------------------
    def bestPath(self, agent):

        # We keep track of the minimum length of the path and the bestpath
        minlen = math.inf
        bestpath = []

        # For every gate in the Skynet
        for gate in self.gateways.keys():

            # Obtaning all the paths and choosing the one driving to the agent
            tablenodes = self.shortestPaths(gate)
            path = tablenodes[agent][1].split(";")[:-1]

            # If the calculated value of the length of the path is lower...
            # ...taking into account the gateway logic in "shortestPaths"
            if tablenodes[agent][0] < minlen:
                minlen = tablenodes[agent][0]
                closestgate = gate
                bestpath = path
            #If it is equal, we compare the actual length of the path
            elif tablenodes[agent][0] == minlen:
                if len(path) < len(bestpath):
                    minlen = tablenodes[agent][0]
                    closestgate = gate
                    bestpath = path

        # we delete the link (both ways) so it is not considered in next turns as a valid link
        del self.nodes[int(bestpath[0])].links[int(bestpath[1])]
        del self.nodes[int(bestpath[1])].links[int(bestpath[0])]

        return bestpath

    def __str__(self):

        string = "Skynet size:" + str(self.size) + "\n\n"
        for node in self.nodes.values():
            string = string + str(node) + "\n"

        string = string + "\nSkynet Gateways: " + str(self.numgateways) + " => " + str(self.gateways)
        return string


class Node:

    def __init__(self, key):
        self.key = key
        self.links = {}

    def __str__(self):
        string = str(self.key) + " linked to: "
        for i in self.links:
            string = string + str(i) + ","

        return (string[:-1])


# --------------- Main program -----------------------


numnodes, numlinks, numgates = [int(i) for i in input().split()]

# We create the Skynet network
mySky = Skynet(numnodes, numlinks, numgates)

# We add all links to the Skynet
for i in range(numlinks):
    mySky.addLink([int(j) for j in input().split()])

# We add al gateways to the Skynet
for i in range(numgates):
    mySky.addGateway(int(input()))

# game loop
while True:
    # Find the best path for the node on which the Skynet agent is positioned this turn (int(input()))
    thepath = mySky.bestPath(int(input()))
    print(thepath[0]+" "+thepath[1])

# 4 4 1
# 0 1
# 0 2
# 1 3
# 2 3
# 3
