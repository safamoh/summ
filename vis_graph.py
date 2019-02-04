import igraph
from random import randint
from graph_utils import calc_rank_percent_distribution

def _plot(topic_id,g, membership=None):
    print ("CREATING VISUALIZATION...")
    print ("Number of clusters: "+str(len(membership)))
    
    #Convert weights to distributed 0..100 for gradient view
    distributed=calc_rank_percent_distribution(g.es["weight"])
    distributed = [int(x * 100) for x in distributed] #
    distributed=map(lambda x: 0 if x<90 else x, distributed) #Modify only show top

    
    #Create lookup for clusters
    id2cluster={}
    for cluster_num,members in enumerate(membership):
        for idx in members:
            id2cluster[idx]=cluster_num

    if membership is not None:
        gcopy = g.copy()
        edges = []
        edges_colors = []
        for edge in g.es():
            #If not in same cluster then grey
            if id2cluster[edge.tuple[0]] != id2cluster[edge.tuple[1]]:
                edges.append(edge)
                edges_colors.append("gray")
            else:
                edges_colors.append("black")
        gcopy.delete_edges(edges)
        
        layout = gcopy.layout("kk")
        g.es["color"] = edges_colors
    else:
        g.es["color"] = "gray"

    visual_style = {}
    visual_style["vertex_label_dist"] = 0
    visual_style["vertex_shape"] = "circle"
    visual_style["edge_color"] = g.es["color"]
    visual_style["vertex_size"] = 30
    # visual_style["bbox"] = (4000, 2500)
    visual_style["bbox"] = (1024, 768)
    visual_style["margin"] = 40
    
    #Special adjustments
    visual_style["edge_curved"] = False  # True
    #option    visual_style['edge_width'] = [w for w in g.es['weight']]


    #Edge colors
    num_colors = 200  # 100-> causes color index too large

    #> this assigns based on degree
    #degree    num_colors = max(gcopy.degree()) + 1
    #    palette = igraph.RainbowPalette(n=num_colors) #Hue
    #RainbowPalette(n=120, s=1, v=0.5, alpha=0.75)
    #    palette = igraph.GradientPalette('white','black',n=num_colors)
    #    palette = igraph.GradientPalette((1.0, 1.0, 1.0, 0.0),'black',n=num_colors) #4th is opacity
    #ok    palette = igraph.GradientPalette((1.0, 1.0, 1.0, 0.0),(0,0,0,1),n=num_colors) #4th is opacity
    #                                  R     G   B         R   G  B 

    
    #Note: at 0 opacity since filter weights to 0 most wouldn't show
    palette = igraph.GradientPalette((1, 1, 1, 0),(0.3,0.3,0.3,1),n=num_colors) #4th is opacity
    
    print ("LENGTH DIS: "+str(len(distributed)))
    color_list = [palette.get(weight_dist) for weight_dist in distributed]  #If fails:  increase num_colors
    visual_style["edge_color"] = color_list


    ##  DEFINE LAYOUT
    layout=''
    #Standard 1)
    #- tight clusters ok
    layout_name="kk"
    
    #Standard 2)
    #About same as 1
    #layout = gcopy.layout_fruchterman_reingold(weights=gcopy.es["weight"], maxiter=1000)#, area=N ** 3, repulserad=N ** 3)
    #?need more    visual_style["layout"] = g.layout_fruchterman_reingold(weights=g.es["weight"], maxiter=1000)

    #Standard 3)
    #- Supposed to be that circle thing
    #- more overlap but decent
    #layout_name="fr"
    
    #Standard 4)
    #- circular edge
    #- VERY noise as all to edge
    #layout=gcopy.layout_circle()

    if layout:
        pass
    else:
        layout = gcopy.layout(layout_name)

    visual_style["layout"] = layout


    for vertex in g.vs():
        vertex["label"] = vertex.index

    if membership is not None:
        colors = []
        for i in range(0, len(membership)+1):
            colors.append('%06X' % randint(0, 0xFFFFFF))
        for vertex in g.vs():
#            vertex["color"] = str('#') + colors[membership[vertex.index]]
            vertex["color"] = str('#') + colors[id2cluster[vertex.index]]
        visual_style["vertex_color"] = g.vs["color"]

    print ("plotting...")
    
    #Include filename if not to screen
    out=igraph.plot(g, **visual_style)
    out.save("./plots/"+topic_id + '_plot.png')

    return


if __name__ == "__main__":
    #fails# g = igraph.Nexus.get("karate")

    #cl = g.community_fastgreedy()
    #membership = cl.as_clustering().membership

    _plot(g, membership)